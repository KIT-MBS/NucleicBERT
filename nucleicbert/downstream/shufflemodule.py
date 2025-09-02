import torch
import torch.nn as nn
import pytorch_lightning as pl
import typing
import torchmetrics as tm
from transformers import get_cosine_schedule_with_warmup
import pandas as pd

from nucleicbert.models.bert import BERT


class ShuffleDetector(nn.Module):
    def __init__(
            self,
            pretrained_model,
            hidden_dim=1024,
            num_layers=32,
            num_heads=32,
        ):
        super().__init__()
        self.pretrained_model = pretrained_model
        

        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
            
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),  
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    
    def forward(self, input_ids):

        pretrained_output = self.pretrained_model(input_ids, output_attentions=True)
        

        input_list = pretrained_output[2]  # list of [B, 1, L, Hidden_Dim]
        

        last_hidden_state = input_list[-1]  # last layer's hidden state [B, 1, seq_len, Hidden_Dim]
        last_hidden_state = last_hidden_state.squeeze(1)  # [B, seq_len, Hidden_Dim]
        

        attention_mask = (input_ids != 0).float()  # [B, seq_len]
        

        attention_scores = self.attention_pool(last_hidden_state)  # [B, seq_len, 1]
        

        attention_scores = attention_scores.squeeze(-1)  # [B, seq_len]

        attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e10)
        

        attention_weights = nn.functional.softmax(attention_scores, dim=1)  # [B, seq_len]
        

        weighted_sum = torch.bmm(
            attention_weights.unsqueeze(1),  # [B, 1, seq_len]
            last_hidden_state  # [B, seq_len, hidden_dim]
        )  # Result: [B, 1, hidden_dim]
        
        weighted_sum = weighted_sum.squeeze(1)  # [B, hidden_dim]
        

        output = self.classifier(weighted_sum)  # [B, 1]
        
        return output
    
class ShuffleDetectorModule(pl.LightningModule):
        def __init__(
            self,
            model_config: dict, 
            run_config: dict,
            tokenizer,
        ) -> None:
            super(ShuffleDetectorModule, self).__init__()
            self.tokenizer = tokenizer
            self.save_hyperparameters()

            self.use_scheduler = run_config.get('use_scheduler', True)  
            self.max_epochs = run_config['max_epochs']
            self.lr = run_config.get('lr', 5e-5)  # Lower learning rate
            self.weight_decay = run_config.get('weight_decay', 0.01)
            self.internal_logs_freq = run_config['internal_logs_freq']
            self.warmup_steps = run_config.get('warmup_steps', int(0.1 * self.max_epochs))
            

            self.vocab_size = model_config['vocab_size']
            self.hidden_size = model_config['hidden_size']
            self.num_hidden_layers = model_config['num_hidden_layers']
            self.num_attention_heads = model_config['num_attention_heads']
            self.dropout = model_config['dropout']
            self.max_length = model_config['max_length']
            self.position_embedding = model_config['position_embedding']

            self.bert = BERT(
                vocab_size=self.vocab_size,
                hidden_size=self.hidden_size,
                num_hidden_layers=self.num_hidden_layers,
                num_attention_heads=self.num_attention_heads,
                dropout=self.dropout,
                max_length=self.max_length,
                position_embedding=self.position_embedding,
            )
            if run_config['pretrained_path']:
                self.bert.load_state_dict(torch.load(run_config['pretrained_path'], map_location=self.device))
                #! be careful here, if the models don't have same names, it will throw an error
                #! here I use strict=False, so that unncessary weights for Pytorch-Lightning are not loaded

            self.model = ShuffleDetector(
                pretrained_model=self.bert,
                hidden_dim=self.hidden_size,
                num_layers=self.num_hidden_layers,
                num_heads=self.num_attention_heads,
            )


            self._init_weights(self.model.classifier)
            self._init_weights(self.model.attention_pool)

            if run_config['frozen']:
                print('Freezing the pretrained model.')
                self.freeze_pretrained_layers()

            self.results_dict = {
                'sequences': [],
                'predictions': [],
                'targets': [],
                'names': [],
            }
                
        def _init_weights(self, module):
            """Initialize the weights for better convergence"""
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()

        def forward(self, input_ids):
            return self.model(input_ids)
        
        def training_step(
            self, 
            batch: typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str], 
            batch_idx: int
        ) -> torch.Tensor:
            input_ids, target, seq, name = batch
            logits = self.forward(input_ids)
            
            # Convert target to float for BCE loss
            target = target.float()
            
            loss = self.loss(logits, target)
            self.log("Training Loss", loss, sync_dist=True, prog_bar=True)
            acc, prec, f1, recall = self.metrics(logits, target)
            self.log("Training Accuracy", acc, sync_dist=True, prog_bar=True)
            self.log("Training Precision", prec, sync_dist=True)
            self.log("Training F1 Score", f1, sync_dist=True)
            self.log("Training Recall", recall, sync_dist=True)

            return loss
    
        def validation_step(
                self, 
                batch: typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str], 
                batch_idx: int
            ) -> torch.Tensor:
            input_ids, target, seq, name = batch
            logits = self.forward(input_ids)
            

            target = target.float()
            
            loss = self.loss(logits, target)
            self.log("Validation Loss", loss, sync_dist=True, prog_bar=True)
            acc, prec, f1, recall = self.metrics(logits, target)
            self.log("Validation Accuracy", acc, sync_dist=True, prog_bar=True)
            self.log("Validation Precision", prec, sync_dist=True)
            self.log("Validation F1 Score", f1, sync_dist=True)
            self.log("Validation Recall", recall, sync_dist=True)
            

            if hasattr(self, 'logger') and self.logger is not None:
                if self.global_rank == 0 and (self.current_epoch+1) % self.internal_logs_freq == 0:

                    logits_cpu = logits.detach().cpu()
                    target_cpu = target.detach().cpu()
                    

                    predictions = torch.sigmoid(logits_cpu).numpy()
                    predictions_binary = (predictions > 0.5).astype(int).flatten().tolist()
                    targets_list = target_cpu.numpy().flatten().tolist()
                    

                    self.results_dict['sequences'].extend([str(s) for s in seq])
                    self.results_dict['predictions'].extend(predictions_binary)
                    self.results_dict['targets'].extend(targets_list)
                    self.results_dict['names'].extend(name)

            return loss
        
        def test_step(
            self, 
            batch: typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str], 
            batch_idx: int
        ) -> torch.Tensor:
            input_ids, target, seq, name = batch
            logits = self.forward(input_ids)
            

            target = target.float()
            
            loss = self.loss(logits, target)
            self.log("Test Loss", loss, sync_dist=True, prog_bar=True)
            acc, prec, f1, recall = self.metrics(logits, target)
            self.log("Test Accuracy", acc, sync_dist=True, prog_bar=True)
            self.log("Test Precision", prec, sync_dist=True)
            self.log("Test F1 Score", f1, sync_dist=True)
            self.log("Test Recall", recall, sync_dist=True)

            return loss
        
        def on_validation_epoch_end(self):
            """Log results at the end of each validation epoch"""

            if (self.current_epoch+1) % self.internal_logs_freq == 0 and hasattr(self, 'logger') and self.logger is not None:
                if len(self.results_dict['predictions']) > 0:
                    try:

                        results_df = pd.DataFrame({
                            'sequence': self.results_dict['sequences'],
                            'prediction': self.results_dict['predictions'],
                            'target': self.results_dict['targets'],
                            'name': self.results_dict['names']
                        })
                        

                        if hasattr(self.logger, 'experiment') and hasattr(self.logger.experiment, 'log'):
                            import wandb

                            table = wandb.Table(dataframe=results_df)
                            self.logger.experiment.log({"validation_results": table})
                            

                        self.results_dict = {
                            'sequences': [],
                            'predictions': [],
                            'targets': [],
                            'names': [],
                        }
                    except Exception as e:

                        print(f"Error logging to WandB: {e}")
             
        def loss(
            self, 
            logits: torch.Tensor, 
            target: torch.Tensor
        ) -> torch.Tensor:

            logits = logits.squeeze(dim=-1)
            

            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, target)
            return loss
    
        def metrics(self, logits, target):
            accuracy_fn = tm.classification.BinaryAccuracy().to(self.device)
            precision_fn = tm.classification.BinaryPrecision().to(self.device)
            f1score_fn = tm.classification.BinaryF1Score().to(self.device)
            recall_fn = tm.classification.BinaryRecall().to(self.device)
            

            logits = logits.squeeze(dim=-1)
            
            acc = accuracy_fn(logits, target)
            prec = precision_fn(logits, target)
            f1 = f1score_fn(logits, target)
            recall = recall_fn(logits, target)
            return acc, prec, f1, recall
        
        def configure_optimizers(self):
            pretrained_params = []
            classifier_params = []
            
            for name, param in self.named_parameters():
                if 'bert' in name and param.requires_grad:
                    pretrained_params.append(param)
                else:
                    classifier_params.append(param)
            
            optimizer = torch.optim.AdamW([
                {'params': pretrained_params, 'lr': self.lr * 0.1},  # Lower LR for pretrained
                {'params': classifier_params, 'lr': self.lr}
            ], weight_decay=self.weight_decay)

            if self.use_scheduler:
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=self.warmup_steps,
                    num_training_steps=self.max_epochs,
                )
                return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
            else:
                return optimizer
            
        def freeze_pretrained_layers(self) -> None:
            for param in self.bert.parameters():
                param.requires_grad = False