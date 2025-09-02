import torch
import torch.nn as nn
import pytorch_lightning as pl
import typing
import torchmetrics as tm
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import time

from nucleicbert.models.bert import BERT
from nucleicbert.pretrain.utils import Config


torch.set_float32_matmul_precision('high')
# enable flash attention
torch.backends.cuda.enable_flash_sdp(enabled=True)




class PreTrainingModule(pl.LightningModule):
    """
    A pytorch lightning module for pretraining BERT.

    Args:
        config: A ConfigUtils object containing the configuration parameters.

    Attributes:
        model: A BERT object.
        lr: The learning rate.
        loss_type: The type of loss to use. Can be 'normal' or 'custom'.
        loss_fn: The loss function to use.
    
    """
    def __init__(
            self,
            model_config,
            run_config: typing.Dict[str, typing.Any],
            tokenizer
        ) -> None:
        super(PreTrainingModule, self).__init__()
        self.run_config = run_config if isinstance(run_config, Config) else run_config
        self.model_config = model_config
        self.model = BERT(**self.model_config)
        self.lr = self.run_config.get('lr')
        self.use_scheduler = self.run_config.get('use_scheduler')
        self.max_epochs = self.run_config.get('max_epochs')
        self.weight_decay = self.run_config.get('weight_decay')
        self.internal_metrics_log_freq = self.run_config.get('internal_metrics_log_freq')

        self.tokenizer = tokenizer
        self.loss_fn = nn.CrossEntropyLoss(ignore_index = -100)

        self.precision_fn = tm.classification.MulticlassPrecision(
            num_classes = self.model_config['vocab_size'],
            ignore_index = -100,
        )
        self.accuracy_fn = tm.classification.MulticlassAccuracy(
            num_classes = self.model_config['vocab_size'],
            ignore_index = -100,
        )
        self.f1_score_fn = tm.classification.MulticlassF1Score(
            num_classes = self.model_config['vocab_size'],
            ignore_index = -100,
        )
        self.confusion_matrix_fn = tm.classification.MulticlassConfusionMatrix(
            num_classes = self.model_config['vocab_size'],
            ignore_index = -100,
            normalize = 'true',
        )
        

        if self.run_config.get('init_weights', False):
            self.apply_weight_initialization()

        # used for saving model outputs, used for testing the code
        self.output = {
            'Input IDs': [],
            'Predictions': [],
            'Target IDs': [],
        }
    

    def forward(self, input_ids) -> torch.Tensor:
        """
        The forward function of the module.

        Args:
            input_ids: A tensor of shape (batch_size, max_length) containing the input token ids.
            masked_positions: A tensor of shape (batch_size, max_pred_per_seq) containing the positions of the masked tokens.

        Returns:
            A tuple containing the output given by BERT in pretraining mode, which gives the logits as output.
        

        """
        logits = self.model.forward(input_ids)
        return logits
        
    def training_step(self, batch:typing.Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        input_ids, target_ids = batch[0], batch[1]
        logits = self.forward(input_ids) # logits shape [batch_size, max_length, vocab_size]
        loss = self.loss(logits, target_ids)
        acc, prec, f1_score = self.metrics(logits, target_ids)
        self.log('Training Loss', loss, sync_dist = True, prog_bar = True)
        self.log('Training Accuracy', acc, sync_dist = True, prog_bar = True)
        self.log('Training Precision', prec, sync_dist = True)
        self.log('Training F1 Score', f1_score, sync_dist = True)
        

        if hasattr(self, 'scheduler'):
            self.log('Learning Rate', self.scheduler.get_last_lr()[0], sync_dist = True, prog_bar = True)
            
        return loss
    
    def validation_step(self, batch: typing.Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        input_ids, target_ids = batch[0], batch[1]
        logits = self.forward(input_ids)
        loss = self.loss(logits, target_ids)
        acc, prec, f1_score = self.metrics(logits, target_ids)
        self.log('Validation Loss', loss, sync_dist = True, prog_bar = True)
        self.log('Validation Accuracy', acc, sync_dist = True, prog_bar = True)
        self.log('Validation Precision', prec, sync_dist = True)
        self.log('Validation F1 Score', f1_score, sync_dist = True)
        if self.global_rank==0 and (self.current_epoch+1)%self.internal_metrics_log_freq == 0:
            preds_for_mask_pos, target_ids_for_mask_pos = self._get_model_pred_and_target(logits, target_ids)
            self.input_ids = input_ids[0].tolist()
            self.preds_for_mask_pos = preds_for_mask_pos
            self.target_ids_for_mask_pos = target_ids_for_mask_pos
            preds = self._get_pred_from_logits(logits)
            self.cm = self.confusion_matrix_fn(preds, target_ids)
        return loss
    
    def on_train_epoch_end(self):
        if self.global_rank==0 and (self.current_epoch+1)%self.internal_metrics_log_freq == 0:
            self.output['Input IDs'].append(str(self.input_ids))
            self.output['Predictions'].append(str(self.preds_for_mask_pos))
            self.output['Target IDs'].append(str(self.target_ids_for_mask_pos))
            image = self.cm_plot(self.cm)
            self.logger.log_image(key = 'Confusion Matrix', images = [image])


    def on_train_start(self):
        self.start_time = time.time()
        
    
    def on_train_end(self):
        #! if training stops in the middle, the outputs will not be logged
        if self.global_rank==0:
            end_time = time.time()
            runtime = round(end_time - self.start_time, 3)
            print(f"Runtime: {runtime}")
            dataframe = pd.DataFrame(self.output)
            self.logger.log_text(key='Outputs', dataframe=dataframe)


    def loss(self, logits, target_ids):
        loss = self.loss_fn(logits.view(-1, self.model_config['vocab_size']), target_ids.view(-1))
        return loss
    
    def metrics(self, logits, target_ids):
        preds = self._get_pred_from_logits(logits)
        acc = self.accuracy_fn(preds, target_ids)
        prec = self.precision_fn(preds, target_ids)
        f1_score = self.f1_score_fn(preds, target_ids)
        return acc, prec, f1_score
    
    def _get_pred_from_logits(self, logits):
        _, topk_idx = torch.topk(torch.softmax(logits, dim=-1), dim = -1, k=1)
        preds = topk_idx.squeeze(dim=-1)
        return preds
    

    def _get_model_pred_and_target(self, logits, target_ids):
        logits = logits[0]
        target_ids = target_ids[0]
        preds = self._get_pred_from_logits(logits)
        masked_pos = (target_ids != -100)
        preds = preds[masked_pos]
        target_ids = target_ids[masked_pos]
        preds = preds.squeeze().tolist()
        target_ids = target_ids.squeeze().tolist()
        if type(preds) != list:
            preds = [preds]
        if type(target_ids) != list:
            target_ids = [target_ids]
        return preds, target_ids
    

    def cm_plot(self, cm):
        add_text = True
        dpi = 300
        labelsize = 30
        if cm.size(0) > 100:
            warnings.warn('Size of vocab is too big, the confusion matrix matrix will not be helpful for analysis.')
            add_text = False
            dpi = 100
            labelsize = 10
        fig, ax = self.confusion_matrix_fn.plot(val = cm, add_text=add_text)
        fig.set_size_inches(20, 20)
        ax.set_title(f'Confusion Matrix', fontsize=30, fontweight='bold')
        ax.set_xlabel('Predicted Class', fontsize=30, fontweight='bold')
        ax.set_ylabel('True Class', fontsize=30, fontweight='bold')
        ax.tick_params(axis='both', labelsize=labelsize)
        fig.set_dpi(dpi)
        # if the vocab size is too big, the consufion matrix will be too big to analyze, but this will be useful for small vocab size
        return fig
    

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
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




    def apply_weight_initialization(self):
        """
        Apply weight initialization to the entire BERT model.
        You can customize the initialization method based on your preference.
        """
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                if param.ndimension() > 1:
                    # Apply weight initialization to weight tensors
                    torch.nn.init.xavier_uniform_(param.data)



if __name__ == '__main__':
    import nucleicbert.tokenization as tokenization
    from torch.utils.data import DataLoader
    from dataset import RNASeqDataset
    from models import BERT

    min_length = 0
    max_length = 10
    input_dir = "../data/synthetic_data/train_single/"
    config = {
        'max_length': max_length,
        'min_length': min_length,
        'input_dir': input_dir,
        'positional_embedding': 'sinusoidal',
    }
    tokenizer = tokenization.FullTokenizer(k=1)
    dataset = RNASeqDataset(config, tokenizer)
    dataloader = DataLoader(dataset, shuffle = True)
    model_config = {}
    run_config = {}
    model = BERT()
    model_final = PreTrainingModule(model_config, run_config)
    trainer = pl.Trainer(max_epochs = 1)
    trainer.fit(model_final, dataloader)
