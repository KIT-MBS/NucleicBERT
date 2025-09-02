import pytorch_lightning as pl
import torch
import torch.nn as nn
import typing
import torchmetrics as tm
from transformers import get_cosine_schedule_with_warmup
import pandas as pd

from nucleicbert.models.bert import BERT


class BERTWithFitnessPredictor(nn.Module):
    def __init__(self, bert, hidden_size):
        super(BERTWithFitnessPredictor, self).__init__()
        self.bert = bert
        self.fitness_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)  # Single output for fitness value
        )

    def forward(self, input_ids):
        x,_,_, = self.bert.encoder(input_ids, need_weights=True)
        x = x[:, 0] # [batch_size, embed_dim]
        output = self.fitness_head(x) 
        return output

class FitnessPredictor(pl.LightningModule):
    """
    A pytorch lightning module for downstream training of BERT.

    Args:
        config: A ConfigUtils object containing the configuration parameters.

    Attributes:
        bert: A BERT object.
        lr: The learning rate.
        loss_fn: The loss function to use.

    If there is no pretrained model, the model is initialized with random weights.
    Backbone of the pretrained model can be frozen by setting the frozen parameter to True in the config file.
    
    """

    def __init__(
            self,
            model_config: dict, 
            run_config: dict,
            tokenizer,
        ) -> None:
        super(FitnessPredictor, self).__init__()
        self.tokenizer = tokenizer

        self.use_scheduler = run_config.get('use_scheduler', False)
        self.max_epochs = run_config['max_epochs']
        self.lr = run_config['lr']
        self.weight_decay = run_config['weight_decay']
        self.internal_logs_freq = run_config['internal_logs_freq']
        

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
        if run_config['pretrained_path']:
            self.bert.load_state_dict(torch.load(run_config['pretrained_path'], map_location=self.device))
            #! be careful here, if the models don't have same names, it will throw an error
            #! here I use strict=False, so that unncessary weights for Pytorch-Lightning are not loaded

        self.model = BERTWithFitnessPredictor(
            self.bert,
            hidden_size,
        )


        if run_config['frozen']:
            print('Freezing the pretrained model.')
            self.freeze_pretrained_layers()

        self.output = {
            'pred_fitness': [],
            'sequence': [],
            'num_mutations': [],
            'target_fitness': [],
            'epoch': [],
        }


    def forward(
        self, 
        input_ids: torch.Tensor, 
    )-> torch.Tensor:
        """
        The forward function of the module.

        Args:
            input_ids: A tensor of shape (batch_size, max_length) containing the input token ids.
            seq_len: A tensor of shape (batch_size) containing the sequence lengths of the inputs.

        Returns:
            A tensor of shape (batch_size, seq_len, seq_len) containing the contact map predictions.

        It uses the encoder of the pretrained model to get the output.
        In downstream mode, the encoder returns a tuple containing a list of attention weights and a list of embeddings.
        The downstream model takes the attention weights or embeddings as input and returns the predictions.

        """
        output = self.model(input_ids)
        return output
    
    
    def training_step(
            self, 
            batch: typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str], 
            batch_idx: int
        ) -> torch.Tensor:
        input_ids, target, seq, num_mutations = batch
        logits = self.forward(input_ids)
        loss = self.loss(logits, target)
        self.log("Training Loss", loss, sync_dist = True, prog_bar = True)
        r2score, mse = self.metrics(logits, target)
        self.log("Training R2 Score", r2score, sync_dist = True, prog_bar = True)
        self.log("Training MSE", mse, sync_dist = True)

        return loss
    
    def validation_step(
            self, 
            batch: typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str], 
            batch_idx: int
        ) -> torch.Tensor:
        input_ids, target, seq, num_mutations = batch
        logits = self.forward(input_ids)
        loss = self.loss(logits, target)
        self.log("Validation Loss", loss, sync_dist = True, prog_bar = True)
        r2score, mse = self.metrics(logits, target)
        self.log("Validation R2 Score", r2score, sync_dist = True, prog_bar = True)
        self.log("Validation MSE", mse, sync_dist = True)
        if self.global_rank == 0 and (self.current_epoch+1)%self.internal_logs_freq == 0:
            # Fix: Handle batch data correctly
            batch_size = len(target)
            self.output['pred_fitness'].extend(logits.squeeze(-1).cpu().detach().numpy().tolist())
            self.output['sequence'].extend([str(s) for s in seq])
            self.output['num_mutations'].extend(num_mutations.cpu().detach().numpy().tolist())
            self.output['target_fitness'].extend(target.cpu().detach().numpy().tolist())
            self.output['epoch'].extend([self.current_epoch+1] * batch_size)

        return loss

    def _log_output(self):
        if self.global_rank == 0 and self.output['pred_fitness']:
            df = pd.DataFrame(self.output)
            # Fix: Use proper logging method
            self.logger.log_text(key='Validation Predictions', dataframe=df)
            self.output = {
                'pred_fitness': [],
                'sequence': [],
                'num_mutations': [],
                'target_fitness': [],
                'epoch': [],
            }
    
    def test_step(
            self, 
            batch: typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str], 
            batch_idx: int
        ) -> torch.Tensor:
        input_ids, target, seq, num_mutations = batch
        logits = self.forward(input_ids)
        loss = self.loss(logits, target)
        self.log("Test Loss", loss, sync_dist = True, prog_bar = True)
        r2score, mse = self.metrics(logits, target)
        self.log("Test R2 Score", r2score, sync_dist = True, prog_bar = True)
        self.log("Test MSE", mse, sync_dist = True)

        return loss

    def on_fit_end(self):
        self._log_output()
    
    def loss(
            self, 
            logits: torch.Tensor, 
            target: torch.Tensor
        ) -> torch.Tensor:

        logits = logits.squeeze(-1)
        loss_fn = nn.L1Loss()
        loss = loss_fn(logits, target)
        return loss
    
    def metrics(self, logits, target):
        logits = logits.squeeze(-1)
        r2score_fn = tm.R2Score().to(self.device)
        mse_fn = tm.MeanSquaredError().to(self.device)
        r2score = r2score_fn(logits, target)
        mse = mse_fn(logits, target)
        return r2score, mse


    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        if self.use_scheduler:
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=0.1 * self.max_epochs,
                num_training_steps=self.max_epochs,
            )
            return [self.optimizer], [self.scheduler]
        else:
            return [self.optimizer]
        

    def freeze_pretrained_layers(self) -> None:
        for param in self.bert.parameters():
            param.requires_grad = False