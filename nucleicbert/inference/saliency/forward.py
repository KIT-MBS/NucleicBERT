import torch
import torch.nn as nn
import torch.nn.functional as F
from nucleicbert.utils.focal_loss import FocalLossMultiClass

class SaliencyForward:
    """
    Class that handles different forward passes for saliency map generation
    based on different model architectures.
    """
    def __init__(
            self,
            model: nn.Module,
            model_type: str,
            device: str = 'cuda',
        ):
        """
        Initialize the saliency forward pass class
        
        Args:
            model: The model to generate saliency maps for
            model_type: The type of model ('bert', 'bert_with_secstr', 'bert_with_closeness', 'bert_with_contactmap')
            device: Device to run the model on
        """
        self.model = model
        self.model_type = model_type
        self.device = device
        
        if model_type == 'bert':
            self.forward_fn = self.bert_saliency_forward
            self.loss_fn = nn.CrossEntropyLoss(
                ignore_index=-100,
                reduction='mean',
            )
        elif model_type == 'bert_with_secstr':
            self.forward_fn = self.bert_saliency_forward_with_secstr
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif model_type == 'bert_with_closeness':
            self.forward_fn = self.bert_saliency_forward_with_closeness
            self.loss_fn = FocalLossMultiClass(
                alpha=0.8,
                gamma=2,
                ignore_index=-1,
            )
        elif model_type == 'bert_with_shuffle':
            self.forward_fn = self.bert_saliency_forward_with_shuffle
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def bert_saliency_forward(
            self,
            input_ids: torch.Tensor,
            target_ids: torch.Tensor,
        ) -> torch.Tensor:
        """
        Generate saliency map for basic BERT model (MLM pretraining)
        
        Args:
            input_ids: Input token IDs
            target_ids: Target token IDs for MLM
            
        Returns:
            Normalized saliency map tensor
        """
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)
        input_ids = input_ids[:, 1:-1]  # Remove CLS and SEP tokens
        target_ids = target_ids[:, 1:-1]  # Remove CLS and SEP tokens

        # Get token embeddings
        input_embd = self.model.encoder.embedding.token(input_ids)
        
        # Zero gradients before forward pass
        self.model.zero_grad()
        
        # Forward pass
        position_embd = self.model.encoder.embedding.position(input_embd)
        input_embd = input_embd + position_embd
        hidden_states = self.model.encoder.embedding.dropout(input_embd)
        attention_mask = input_ids.ne(0).unsqueeze(2)
        attention_mask = attention_mask.to(dtype=torch.float32)
        
        encoder_output = self.model.encoder.encoder_forward(hidden_states, attention_mask, need_weights=True)
        outputs = self.model.mlm(encoder_output[0])
        attn_weights = torch.cat(encoder_output[1], dim=1)
        
        # Compute loss and gradients
        loss = self.loss_fn(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
        input_embd.retain_grad()
        loss.backward()
        
        # Compute saliency map
        saliency_map = input_embd.grad.abs().mean(dim=-1)
        max_val = saliency_map.max()
        if max_val > 0:
            saliency_map = saliency_map / max_val

        return saliency_map, attn_weights

    def bert_saliency_forward_with_secstr(
            self,
            input_ids: torch.Tensor,
            target_ids: torch.Tensor,
        ) -> torch.Tensor:
        """
        Generate saliency map for BERT with secondary structure prediction
        
        Args:
            input_ids: Input token IDs
            target_ids: Target secondary structure labels (2D pairing matrix)
            
        Returns:
            Normalized saliency map tensor
        """
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)
        
        # Get token embeddings
        input_embd = self.model.model.bert.encoder.embedding.token(input_ids)
        input_embd = input_embd.requires_grad_()  # Enable gradient tracking
        
        # Zero gradients before forward pass
        self.model.zero_grad()
        
        # Forward pass
        position_embd = self.model.model.bert.encoder.embedding.position(input_embd)
        input_embd = input_embd + position_embd
        hidden_states = self.model.model.bert.encoder.embedding.dropout(input_embd)

        attention_mask = input_ids.ne(0).unsqueeze(2)
        attention_mask = attention_mask.to(dtype=torch.float32)

        encoder_output = self.model.model.bert.encoder.encoder_forward(hidden_states, attention_mask, need_weights=True)
        input_list = encoder_output[2]  # list of [B, 1, L, Hidden_Dim]
        embeddings = torch.cat(input_list, dim=1)  # [B, Hidden_Layers+1, L, Hidden_Dim]
        embeddings = torch.mean(embeddings, dim=1)  # [B, seq_len, Hidden_Dim]
        embeddings = embeddings[:, 1:-1, :]  # Remove CLS and SEP tokens [B, L-2, Hidden]
        
        # Get secondary structure predictions using the prediction head
        logits = self.model.model.pred_head(embeddings)  # [B, L-2, L-2]
        
        # Calculate loss using proper masking (similar to secstrmodule.py)
        batch_size, seq_len = logits.shape[:2]
        
        # Calculate loss only on upper triangular part to avoid counting pairs twice
        upper_tri_mask = torch.triu(torch.ones(seq_len, seq_len, device=logits.device), diagonal=1).bool()
        
        # Apply attention mask to exclude padded positions
        # Create attention mask for the actual sequence (without CLS/SEP)
        seq_attention_mask = input_ids.ne(0)[:, 1:-1]  # Remove [CLS] and [SEP] from attention mask
        valid_positions = seq_attention_mask.unsqueeze(-1) * seq_attention_mask.unsqueeze(-2)  # [B, L, L]
        
        final_mask = upper_tri_mask.unsqueeze(0) & valid_positions.bool()
        

        loss = self.loss_fn(
            logits[final_mask],
            target_ids[final_mask]
        )
        
        input_embd.retain_grad()
        loss.backward()
        
        attn_weights = torch.cat(encoder_output[1], dim=1)
        # The attention weights should match the input sequence length (including CLS/SEP)
        input_seq_len = input_ids.shape[1]
        attn_weights = attn_weights[:, :, :, :input_seq_len, :input_seq_len]  # [B, Layers, H, L, L]
        
        saliency_map = input_embd.grad.abs().mean(dim=-1)
        max_val = saliency_map.max()
        if max_val > 0:
            saliency_map = saliency_map / max_val

        return saliency_map, attn_weights
    
    def bert_saliency_forward_with_closeness(
            self,
            input_ids: torch.Tensor,
            target_ids: torch.Tensor,
        ) -> torch.Tensor:
        """
        Generate saliency map for BERT with RNA closeness prediction
        
        Args:
            input_ids: Input token IDs
            target_ids: Target closeness labels
            
        Returns:
            Normalized saliency map tensor
        """
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)

        input_ids = input_ids[:, 1:-1]  # Remove CLS and SEP tokens
        
        # Get token embeddings
        input_embd = self.model.bert.encoder.embedding.token(input_ids)

        # Zero gradients before forward pass
        self.model.zero_grad()

        # Forward pass
        position_embd = self.model.bert.encoder.embedding.position(input_embd)
        input_embd = input_embd + position_embd
        hidden_states = self.model.bert.encoder.embedding.dropout(input_embd)

        attention_mask = input_ids.ne(0).unsqueeze(2)
        attention_mask = attention_mask.to(dtype=torch.float32)

        encoder_output = self.model.bert.encoder.encoder_forward(hidden_states, attention_mask, need_weights=True)
        input_list = encoder_output[1]  # list of [B, 1, H, L, L]
        attn_weights = torch.cat(input_list, dim=1)  # [B, Layers, H, L, L]

        outputs = self.model.resnet(attn_weights)

        # Compute loss and gradients
        loss = self.loss_fn(outputs, target_ids)
        input_embd.retain_grad()
        loss.backward()
        
        # Compute saliency map
        saliency_map = input_embd.grad.abs().mean(dim=-1)
        max_val = saliency_map.max()
        if max_val > 0:
            saliency_map = saliency_map / max_val

        return saliency_map, attn_weights
    

    def bert_saliency_forward_with_shuffle(
            self,
            input_ids: torch.Tensor,
            target_ids: torch.Tensor,
        ) -> torch.Tensor:
        """
        Generate saliency map for BERT with shuffle detection
        
        Args:
            input_ids: Input token IDs
            target_ids: Target shuffle labels
            
        Returns:
            Normalized saliency map tensor
        """
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)
        

        
        input_embd = self.model.bert.encoder.embedding.token(input_ids)
        input_embd = input_embd.requires_grad_()  # Enable gradient tracking

        self.model.zero_grad()

        # Forward pass
        position_embd = self.model.bert.encoder.embedding.position(input_embd)
        input_embd = input_embd + position_embd
        hidden_states = self.model.bert.encoder.embedding.dropout(input_embd)

        attention_mask = input_ids.ne(0).unsqueeze(2)
        attention_mask = attention_mask.to(dtype=torch.float32)

        encoder_output = self.model.bert.encoder.encoder_forward(hidden_states, attention_mask, need_weights=True)
        input_list = encoder_output[2]  # list of [B, 1, L, Hidden_Dim]
        

        last_hidden_state = input_list[-1]  # Get last layer's hidden state [B, 1, seq_len, Hidden_Dim]
        last_hidden_state = last_hidden_state.squeeze(1)  # [B, seq_len, Hidden_Dim]
        

        attention_mask = (input_ids != 0).float()  # [B, seq_len]
        seq_len = int(attention_mask.sum(dim=1).item())  # Get actual sequence length for each batch
        
        # Apply attention pooling for better sequence representation
        attention_scores = self.model.model.attention_pool(last_hidden_state)  # [B, seq_len, 1]
        
        # Apply mask to attention scores
        attention_scores = attention_scores.squeeze(-1)  # [B, seq_len]
        # Set padding attention scores to large negative value
        attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e10)
        
        attention_weights = nn.functional.softmax(attention_scores, dim=1)  # [B, seq_len]
        

        weighted_sum = torch.bmm(
            attention_weights.unsqueeze(1),  # [B, 1, seq_len]
            last_hidden_state  # [B, seq_len, hidden_dim]
        )  # Result: [B, 1, hidden_dim]
        
        weighted_sum = weighted_sum.squeeze(1)  # [B, hidden_dim]
        
        outputs = self.model.model.classifier(weighted_sum)  # [B, 1]
        outputs = outputs.squeeze(dim=-1)

        loss = self.loss_fn(outputs, target_ids)
        input_embd.retain_grad()
        loss.backward()

        attn_weights = torch.cat(encoder_output[1], dim=1)
        attn_weights = attn_weights[:, :, :, :seq_len-2, :seq_len-2]  # [B, Layers, H, L, L]

        # Compute saliency map
        saliency_map = input_embd.grad.abs().mean(dim=-1)
        max_val = saliency_map.max()
        if max_val > 0:
            saliency_map = saliency_map / max_val

        return saliency_map, attn_weights