
import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score
from transformers import get_cosine_schedule_with_warmup

from nucleicbert.models.bert import BERT


CANONICAL_PAIRS = ['AU', 'UA', 'GC', 'CG', 'GU', 'UG', 'AT', 'TA', 'GT', 'TG']

def prob_mat_to_sec_struct(probs: np.ndarray, seq: str, threshold: float = 0.5, 
                           allow_nc_pairs: bool = False, allow_sharp_loops: bool = False) -> np.ndarray:
    """Convert probability matrix to secondary structure."""
    seq_len = probs.shape[-1]
    
    allowed_pairs_mask = np.logical_not(np.eye(seq_len, dtype=bool))
    
    if not allow_sharp_loops:
        for i in range(seq_len):
            for j in range(max(0, i-3), min(seq_len, i+4)):
                allowed_pairs_mask[i, j] = False
    
    if not allow_nc_pairs:
        seq = seq.upper().replace('T', 'U')  # Convert to RNA for pairing check
        canonical_mask = np.zeros((seq_len, seq_len), dtype=bool)
        for i in range(min(len(seq), seq_len)):
            for j in range(min(len(seq), seq_len)):
                if i < len(seq) and j < len(seq):
                    pair = seq[i] + seq[j]
                    if pair in CANONICAL_PAIRS:
                        canonical_mask[i, j] = True
        allowed_pairs_mask = np.logical_and(allowed_pairs_mask, canonical_mask)
    
    probs = probs.copy()
    probs[~allowed_pairs_mask] = 0.0
    
    sec_struct = (probs > threshold).astype(int)
    sec_struct = clean_conflicts(sec_struct, probs)
    
    return sec_struct


def clean_conflicts(sec_struct: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Remove conflicting base pairs, keeping highest probability ones."""
    clean_ss = np.zeros_like(sec_struct)
    tmp_probs = probs.copy()
    tmp_probs[sec_struct < 1] = 0.0
    
    while np.sum(tmp_probs > 0.0) > 0:
        i, j = np.unravel_index(np.argmax(tmp_probs), tmp_probs.shape)
        
        tmp_probs[i, :] = tmp_probs[j, :] = 0.0
        tmp_probs[:, i] = tmp_probs[:, j] = 0.0
        
        # Add the pair
        clean_ss[i, j] = clean_ss[j, i] = 1
        
    return clean_ss


def ss_metrics(target_ss: np.ndarray, pred_ss: np.ndarray) -> Tuple[float, float, float]:
    """Calculate precision, recall, and F1 for secondary structure prediction."""
    seq_len = target_ss.shape[-1]
    upper_tri_idcs = np.triu_indices(seq_len, k=1)
    
    precision = precision_score(target_ss[upper_tri_idcs], pred_ss[upper_tri_idcs], zero_division=0.0)
    recall = recall_score(target_ss[upper_tri_idcs], pred_ss[upper_tri_idcs], zero_division=0.0)
    
    if precision + recall < 1e-5:
        f1 = 0.0
    else:
        f1 = (2 * precision * recall) / (precision + recall)
    
    return precision, recall, f1



class ResNetBlock(nn.Module):
    """ResNet block with residual connection."""
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
    def forward(self, x):
        residual = x
        x = self.layer_norm1(x)
        x = self.mlp(x)
        x = self.dropout(x)
        x = x + residual
        x = self.layer_norm2(x)
        return x



class SecStruct2DPredictionHead(nn.Module):
    """Prediction head for 2D pairing matrix."""
    def __init__(self, hidden_dim: int, num_blocks: int = 2, dropout: float = 0.1):
        super().__init__()
        

        self.blocks = nn.ModuleList([
            ResNetBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])
        

        self.row_proj = nn.Linear(hidden_dim, hidden_dim // 2)
        self.col_proj = nn.Linear(hidden_dim, hidden_dim // 2)
        

        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        # x shape: [batch, seq_len, hidden_dim]
        

        for block in self.blocks:
            x = block(x)
        

        row_features = self.row_proj(x)  # [batch, seq_len, hidden_dim//2]
        col_features = self.col_proj(x)  # [batch, seq_len, hidden_dim//2]
        

        batch_size, seq_len, feat_dim = row_features.shape
        
        row_expanded = row_features.unsqueeze(2).expand(-1, -1, seq_len, -1)
        col_expanded = col_features.unsqueeze(1).expand(-1, seq_len, -1, -1)
        

        pair_features = torch.cat([row_expanded, col_expanded], dim=-1)
        

        logits = self.out_proj(pair_features).squeeze(-1)
        

        logits = (logits + logits.transpose(-2, -1)) / 2
        
        return logits


# Combined model
class BERTWithSecStr2D(nn.Module):
    def __init__(self, bert, hidden_size, num_resnet_blocks=2, dropout=0.1):
        super().__init__()
        self.bert = bert
        self.pred_head = SecStruct2DPredictionHead(
            hidden_size, 
            num_blocks=num_resnet_blocks,
            dropout=dropout
        )
    
    def forward(self, input_ids):
        pretrained_output = self.bert(input_ids, output_attentions=True)
        input_list = pretrained_output[2] # list of [B, 1, L, Hidden_Dim]
        embeddings = torch.cat(input_list, dim=1) # [B, Hidden_Layers+1, L, Hidden_Dim]
        # embeddings = embeddings[..., :-1, :] # remove the SEP token
        # embeddings = embeddings[..., 1:, :] # remove the CLS token
        #[B, Hidden_Layers+1, seq_len, Hidden_Dim]
        embeddings = torch.mean(embeddings, dim=1) # [B, seq_len, Hidden_Dim]
        embeddings = embeddings[:, 1:-1, :]  # [B, L-2, Hidden]
        
        logits = self.pred_head(embeddings)
        
        return logits



class SecondaryStructure2DPredictor(pl.LightningModule):
    """PyTorch Lightning module for 2D secondary structure prediction."""
    
    def __init__(
        self,
        model_config: dict,
        run_config: dict,
        tokenizer,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        self.tokenizer = tokenizer
        

        self.use_scheduler = run_config.get('use_scheduler', False)
        self.max_epochs = run_config['max_epochs']
        self.lr = run_config['lr']
        self.weight_decay = run_config['weight_decay']
        self.internal_logs_freq = run_config.get('internal_logs_freq', 5)
        

        vocab_size = model_config['vocab_size']
        hidden_size = model_config['hidden_size']
        num_hidden_layers = model_config['num_hidden_layers']
        num_attention_heads = model_config['num_attention_heads']
        dropout = model_config['dropout']
        max_length = model_config['max_length']
        position_embedding = model_config['position_embedding']
        

        self.bert = BERT(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            max_length=max_length,
            position_embedding=position_embedding,
        )
        

        if run_config.get('pretrained_path'):
            print(f"Loading pretrained weights from {run_config['pretrained_path']}")
            self.bert.load_state_dict(torch.load(run_config['pretrained_path'], map_location=self.device))
        

        self.model = BERTWithSecStr2D(
            self.bert,
            hidden_size,
            num_resnet_blocks=run_config.get('num_resnet_blocks', 1),
            dropout=dropout
        )
        

        if run_config.get('frozen', False):
            print('Freezing the pretrained model.')
            self.freeze_pretrained_layers()
        

        self.loss_fn = nn.BCEWithLogitsLoss()
        

        self.threshold = 0.5
        self.threshold_candidates = [i / 100 for i in range(10, 60, 5)]
        self.tune_threshold_every_n_epoch = run_config.get('tune_threshold_every_n_epoch', 1)
        

        self._eval_outputs = defaultdict(lambda: defaultdict(list))
        
    def freeze_pretrained_layers(self) -> None:
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask)
    
    def training_step(self, batch, batch_idx):
        input_ids, target_matrix, _, _, _ = batch
        
        logits = self(input_ids)
        
        batch_size, seq_len = logits.shape[:2]
        

        upper_tri_mask = torch.triu(torch.ones(seq_len, seq_len, device=logits.device), diagonal=1).bool()
        

        # valid_mask = attention_mask[:, 1:-1]  # Remove [CLS] and [SEP]
        # valid_positions = valid_mask.unsqueeze(-1) * valid_mask.unsqueeze(-2)  # [B, L, L]
        

        final_mask = upper_tri_mask.unsqueeze(0) #& valid_positions.bool()
        

        loss = self.loss_fn(
            logits[final_mask],
            target_matrix[final_mask]
        )
        
        self.log("Training Loss", loss, sync_dist=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids, target_matrix, attention_mask, sequences, structures = batch
        
        logits = self(input_ids, attention_mask)
        
        batch_size, seq_len = logits.shape[:2]
        upper_tri_mask = torch.triu(torch.ones(seq_len, seq_len, device=logits.device), diagonal=1).bool()
        
        valid_mask = attention_mask[:, 1:-1]
        valid_positions = valid_mask.unsqueeze(-1) * valid_mask.unsqueeze(-2)
        final_mask = upper_tri_mask.unsqueeze(0) & valid_positions.bool()
        
        loss = self.loss_fn(
            logits[final_mask],
            target_matrix[final_mask]
        )
        
        self.log("Validation Loss", loss, sync_dist=True, prog_bar=True)
        

        if (self.current_epoch + 1) % self.tune_threshold_every_n_epoch == 0:
            self._evaluate_batch(logits, target_matrix, sequences, attention_mask)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        input_ids, target_matrix, attention_mask, sequences, structures = batch
        

        logits = self(input_ids, attention_mask)
        

        batch_size, seq_len = logits.shape[:2]
        upper_tri_mask = torch.triu(torch.ones(seq_len, seq_len, device=logits.device), diagonal=1).bool()
        
        valid_mask = attention_mask[:, 1:-1]
        valid_positions = valid_mask.unsqueeze(-1) * valid_mask.unsqueeze(-2)
        final_mask = upper_tri_mask.unsqueeze(0) & valid_positions.bool()
        
        loss = self.loss_fn(
            logits[final_mask],
            target_matrix[final_mask]
        )
        
        self.log("Test Loss", loss, sync_dist=True, prog_bar=True)
        

        self._evaluate_batch(logits, target_matrix, sequences, attention_mask, 
                           thresholds=[self.threshold], mode='test')
        
        return loss
    
    def _evaluate_batch(self, logits, target_matrix, sequences, attention_mask, 
                       thresholds=None, mode='val'):
        """Evaluate batch with different thresholds."""
        if thresholds is None:
            thresholds = self.threshold_candidates
        
        probs = torch.sigmoid(logits).cpu().numpy()
        target_matrix = target_matrix.cpu().numpy()
        attention_mask = attention_mask.cpu().numpy()
        
        batch_size = probs.shape[0]
        
        for i in range(batch_size):

            actual_len = min(len(sequences[i]), attention_mask[i, 1:-1].sum())
            
            if actual_len > 0:
                prob_mat = probs[i, :actual_len, :actual_len]
                target_mat = target_matrix[i, :actual_len, :actual_len]
                
                for threshold in thresholds:
                    pred_mat = prob_mat_to_sec_struct(
                        prob_mat, sequences[i][:actual_len], 
                        threshold=threshold,
                        allow_nc_pairs=False,
                        allow_sharp_loops=False
                    )
                    
                    precision, recall, f1 = ss_metrics(target_mat, pred_mat)
                    
                    self._eval_outputs[f'{mode}_precision'][threshold].append(precision)
                    self._eval_outputs[f'{mode}_recall'][threshold].append(recall)
                    self._eval_outputs[f'{mode}_f1'][threshold].append(f1)
    
    def on_validation_epoch_end(self):
        if (self.current_epoch + 1) % self.tune_threshold_every_n_epoch == 0:

            best_f1 = -1.0
            best_threshold = self.threshold
            best_metrics = {}
            
            for threshold in self.threshold_candidates:
                if self._eval_outputs['val_f1'][threshold]:
                    avg_f1 = np.mean(self._eval_outputs['val_f1'][threshold])
                    avg_precision = np.mean(self._eval_outputs['val_precision'][threshold])
                    avg_recall = np.mean(self._eval_outputs['val_recall'][threshold])
                    
                    if avg_f1 > best_f1:
                        best_f1 = avg_f1
                        best_threshold = threshold
                        best_metrics = {
                            'f1': avg_f1,
                            'precision': avg_precision,
                            'recall': avg_recall
                        }
            
            self.threshold = best_threshold
            

            if best_metrics:
                self.log("Validation F1", best_metrics['f1'], sync_dist=True, prog_bar=True)
                self.log("Validation Precision", best_metrics['precision'], sync_dist=True)
                self.log("Validation Recall", best_metrics['recall'], sync_dist=True)
                self.log("Best Threshold", self.threshold, sync_dist=True)

            self._eval_outputs.clear()
    
    def on_test_epoch_end(self):

        if self._eval_outputs[f'test_f1'][self.threshold]:
            avg_precision = np.mean(self._eval_outputs[f'test_precision'][self.threshold])
            avg_recall = np.mean(self._eval_outputs[f'test_recall'][self.threshold])
            avg_f1 = np.mean(self._eval_outputs[f'test_f1'][self.threshold])
            
            self.log("Test Precision", avg_precision, sync_dist=True)
            self.log("Test Recall", avg_recall, sync_dist=True)
            self.log("Test F1", avg_f1, sync_dist=True)
            
            print(f"\nTest Results:")
            print(f"  Precision: {avg_precision:.4f}")
            print(f"  Recall: {avg_recall:.4f}")
            print(f"  F1 Score: {avg_f1:.4f}")
            print(f"  Threshold: {self.threshold:.3f}")
        
        self._eval_outputs.clear()
    
    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        
        if self.use_scheduler:
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=int(0.1 * self.max_epochs),
                num_training_steps=self.max_epochs,
            )
            return [self.optimizer], [self.scheduler]
        else:
            return [self.optimizer]
    
    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> List[np.ndarray]:
        """Predict secondary structure as 2D pairing matrix."""
        self.eval()
        with torch.no_grad():
            logits = self(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            predictions = []
            batch_size = probs.shape[0]
            
            for i in range(batch_size):
                if attention_mask is not None:
                    actual_len = attention_mask[i, 1:-1].sum().item()
                    prob_mat = probs[i, :actual_len, :actual_len]
                else:
                    prob_mat = probs[i]
                
                pred_mat = prob_mat_to_sec_struct(
                    prob_mat,
                    seq="N" * prob_mat.shape[0],  # Dummy sequence
                    threshold=self.threshold,
                    allow_nc_pairs=False,
                    allow_sharp_loops=False
                )
                predictions.append(pred_mat)
        
        return predictions
