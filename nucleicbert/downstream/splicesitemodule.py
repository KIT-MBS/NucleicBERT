import pytorch_lightning as pl
import torch
import torch.nn as nn
import typing
import torchmetrics as tm
from transformers import get_cosine_schedule_with_warmup
import pandas as pd

from nucleicbert.models.bert import BERT


class BERTWithSpliceSiteClassifier(nn.Module):
    def __init__(self, bert, hidden_size):
        super(BERTWithSpliceSiteClassifier, self).__init__()
        self.bert = bert
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids):
        x,_,_, = self.bert.encoder(input_ids, need_weights=True)
        x = x[:, 0] # [batch_size, embed_dim]
        output = self.classifier(x) 
        return output

class SpliceSitePredictor(pl.LightningModule):
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
        super(SpliceSitePredictor, self).__init__()
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

        self.model = BERTWithSpliceSiteClassifier(
            self.bert,
            hidden_size,
        )


        if run_config['frozen']:
            print('Freezing the pretrained model.')
            self.freeze_pretrained_layers()

        self.output = {
            'Predictions': [],
            'Target': [],
        }

        self.train_accuracy = tm.classification.BinaryAccuracy()
        self.train_precision = tm.classification.BinaryPrecision()
        self.train_f1 = tm.classification.BinaryF1Score()
        self.train_recall = tm.classification.BinaryRecall()
        
        self.val_accuracy = tm.classification.BinaryAccuracy()
        self.val_precision = tm.classification.BinaryPrecision()
        self.val_f1 = tm.classification.BinaryF1Score()
        self.val_recall = tm.classification.BinaryRecall()
        
        self.test_accuracy = tm.classification.BinaryAccuracy()
        self.test_precision = tm.classification.BinaryPrecision()
        self.test_f1 = tm.classification.BinaryF1Score()
        self.test_recall = tm.classification.BinaryRecall()


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
        input_ids, target, seq= batch
        logits = self.forward(input_ids)
        loss = self.loss(logits, target)
        self.log("Training Loss", loss, sync_dist = True, prog_bar = True)
        
        target_reshaped = target.unsqueeze(dim=-1)
        self.train_accuracy.update(logits, target_reshaped)
        self.train_precision.update(logits, target_reshaped)
        self.train_f1.update(logits, target_reshaped)
        self.train_recall.update(logits, target_reshaped)
        
        self.log("Training Accuracy", self.train_accuracy, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("Training Precision", self.train_precision, on_step=False, on_epoch=True, sync_dist=True)
        self.log("Training F1 Score", self.train_f1, on_step=False, on_epoch=True, sync_dist=True)
        self.log("Training Recall", self.train_recall, on_step=False, on_epoch=True, sync_dist=True)

        return loss
    
    def validation_step(
            self, 
            batch: typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str], 
            batch_idx: int
        ) -> torch.Tensor:
        input_ids, target, seq = batch
        logits = self.forward(input_ids)
        loss = self.loss(logits, target)
        self.log("Validation Loss", loss, sync_dist = True, prog_bar = True)
        
        target_reshaped = target.unsqueeze(dim=-1)
        self.val_accuracy.update(logits, target_reshaped)
        self.val_precision.update(logits, target_reshaped)
        self.val_f1.update(logits, target_reshaped)
        self.val_recall.update(logits, target_reshaped)
        
        self.log("Validation Accuracy", self.val_accuracy, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("Validation Precision", self.val_precision, on_step=False, on_epoch=True, sync_dist=True)
        self.log("Validation F1 Score", self.val_f1, on_step=False, on_epoch=True, sync_dist=True)
        self.log("Validation Recall", self.val_recall, on_step=False, on_epoch=True, sync_dist=True)

        if self.global_rank == 0 and (self.current_epoch+1)%self.internal_logs_freq == 0:
            pred = torch.sigmoid(logits).cpu().numpy()
            target = target.cpu().numpy()
            self.prediction = pred.tolist()
            self.target = target.tolist()

        return loss
    
    def test_step(
            self, 
            batch: typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str], 
            batch_idx: int
        ) -> torch.Tensor:
        input_ids, target, seq = batch
        logits = self.forward(input_ids)
        loss = self.loss(logits, target)
        self.log("Test Loss", loss, sync_dist = True, prog_bar = True)
        
        target_reshaped = target.unsqueeze(dim=-1)
        self.test_accuracy.update(logits, target_reshaped)
        self.test_precision.update(logits, target_reshaped)
        self.test_f1.update(logits, target_reshaped)
        self.test_recall.update(logits, target_reshaped)
        
        self.log("Test Accuracy", self.test_accuracy, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("Test Precision", self.test_precision, on_step=False, on_epoch=True, sync_dist=True)
        self.log("Test F1 Score", self.test_f1, on_step=False, on_epoch=True, sync_dist=True)
        self.log("Test Recall", self.test_recall, on_step=False, on_epoch=True, sync_dist=True)
        
        return loss
    

    def on_train_epoch_end(self):
        if self.global_rank == 0 and (self.current_epoch+1)%self.internal_logs_freq == 0:
            self.output['Predictions'].extend(self.prediction)
            self.output['Target'].extend(self.target)

    def on_fit_end(self):
        self._log_output()

    def _log_output(self):
        if self.global_rank == 0 and self.output['Predictions']:
            dataframe = pd.DataFrame.from_dict(self.output)
            self.logger.log_text(key='Training Predictions', dataframe=dataframe)
            self.output = {
                'Predictions': [],
                'Target': [],
            }

    
    def loss(
            self, 
            logits: torch.Tensor, 
            target: torch.Tensor
        ) -> torch.Tensor:

        target = target.unsqueeze(dim=-1)
        

        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, target)
        return loss
    

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