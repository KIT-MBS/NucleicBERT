import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import typing
import torchmetrics as tm
import matplotlib.pyplot as plt
import matplotlib as mpl
from transformers import get_linear_schedule_with_warmup
import numpy as np
from typing import Optional, Union
import juelich_colors as jc
from matplotlib.colors import LinearSegmentedColormap

from nucleicbert.models.bert import BERT
from nucleicbert.utils.precision_calc import evaluate_prediction, plot_contacts_and_predictions
from nucleicbert.models.resnet import ResidualNetwork2D, ResidualNetwork2DEMBD
from nucleicbert.utils.focal_loss import FocalLossMultiClass
from nucleicbert.utils.contact_map_utils import ContactProbabilityCalculator, DistanceToContactConverter

# enable flash attention
torch.set_float32_matmul_precision('high')
torch.backends.cuda.enable_flash_sdp(enabled=True)

class BERTWithClosenessResNet(nn.Module):
    def __init__(
            self,
            bert,
            input_channels,
            num_residual_blocks,
            downstream_model = 'ResidualNetwork2D',
            task = 'contact_map',
            remove_backbone_width = 0,
        ):
        super(BERTWithClosenessResNet, self).__init__()
        self.bert = bert
        self.downstream_model = downstream_model
        self.task = task
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        is_backbone_class = False
        if remove_backbone_width > 0:
            is_backbone_class = True
        self.cpc = ContactProbabilityCalculator(is_backbone_class=is_backbone_class)
        self.dtcc = DistanceToContactConverter(is_backbone_class=is_backbone_class)

        if self.task == 'contact_map':
            self.num_classes = 2
        elif self.task == 'distance_map':
            self.num_classes = 20
            if remove_backbone_width:
                self.num_classes = 21
        else:
            raise ValueError(f"Task {self.task} not supported.")
        
        if downstream_model == 'ResidualNetwork2D':
            self.resnet = ResidualNetwork2D(
                input_channels=input_channels,
                num_residual_blocks=num_residual_blocks,
                output_channels=self.num_classes
            )
        elif downstream_model == 'ResidualNetwork2DEMBD':
            self.resnet = ResidualNetwork2DEMBD(
                embedding_dim=input_channels,
                num_residual_blocks=num_residual_blocks,
                output_channels=self.num_classes
            )
        else:
            raise ValueError('Downstream model not supported.')
        
    def forward(
            self,
            input_ids: torch.Tensor,
        ) -> torch.Tensor:
        pretrained_output = self.bert(input_ids, output_attentions=True)
        if self.resnet.__class__.__name__ == 'ResidualNetwork2D':
            downstream_input = self.forward_attention_weights(pretrained_output)
        elif self.resnet.__class__.__name__ == 'ResidualNetwork2DEMBD':
            downstream_input = self.forward_embeddings(pretrained_output)

        output = self.resnet(downstream_input)
        return output
    
    def forward_attention_weights(self, pretrained_output: torch.Tensor) -> torch.Tensor:
        input_list = pretrained_output[1] # list of [B, 1, H, L, L]
        assert input_list[0].shape[-1] == input_list[0].shape[-2], "Attention weights should be of shape (batch_size, heads, layers, seq_len, seq_len)"
        attn_weights = torch.cat(input_list, dim=1) # [B, Layers, H, L, L]
        attn_weights = attn_weights[..., :-1, :-1] # remove the SEP token
        attn_weights = attn_weights[..., 1:, 1:] # remove the CLS token
        #[B, Layers, H, seq_len, seq_len]
        return attn_weights
    

    def forward_embeddings(self, pretrained_output: torch.Tensor) -> torch.Tensor:
        input_list = pretrained_output[0] # [B, L, E]
        input_list = input_list[:, 1:-1, :] # remove the SEP and CLS tokens
        #Outer Concatenation
        expand_a = input_list.unsqueeze(2).expand(-1, -1, input_list.size(1), -1) # [B, L, L, E]
        expand_b = input_list.unsqueeze(1).expand(-1, input_list.size(1), -1, -1) # [B, L, L, E]
        embeddings = torch.cat([expand_a, expand_b], dim=-1) # [B, L, L, 2E]
        return embeddings
    
    def predict(
        self, 
        input_ids: torch.Tensor,
        name=None,
    ) -> torch.Tensor:
        """
        Predicts the contact map for a given input.

        Args:
            input_ids: A tensor of shape (batch_size, max_length) containing the input token ids.

        Returns:
            A tensor of shape (batch_size, seq_len, seq_len) containing the contact map predictions.

        """
        input_ids = input_ids.to(self.device)
        logits = self.forward(input_ids)
        if self.task == 'contact_map':
            probs = torch.softmax(logits, dim=1)
            probs = probs[:, 1, :, :]
        elif self.task == 'distance_map':
            probs = self.cpc.compute_detailed_probabilities(logits)['contact_probability']
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        self.plot_blind_contacts(probs.squeeze().cpu().numpy(), ax=ax)
        ax.set_title('Predicted Contacts')
        ax.set_xlabel('Residue Index')
        ax.set_ylabel('Residue Index')
        plt.tight_layout()
        fig.savefig(f'predicted_contacts_{name}.png')
        return probs

class ClosenessPredictor(pl.LightningModule):
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
        super(ClosenessPredictor, self).__init__()
        self.use_scheduler = run_config.get('use_scheduler', False)

        self.max_epochs = run_config['max_epochs']
        self.lr = run_config['lr']
        self.weight_decay = run_config['weight_decay']
        self.internal_log_frequency = run_config['internal_log_frequency']

        self.alpha_focal_loss = run_config.get('alpha_focal_loss', 0.8)
        self.class_weights_list = run_config.get('class_weights')
        self.task = run_config['task']
        remove_backbone_width = run_config.get('remove_backbone_width')
        

        vocab_size = model_config['vocab_size']
        hidden_size = model_config['hidden_size']
        num_hidden_layers = model_config['num_hidden_layers']
        num_attention_heads = model_config['num_attention_heads']
        dropout = model_config['dropout']
        max_length = model_config['max_length']
        position_embedding = model_config['position_embedding']

        downstream_n_layers = model_config['downstream_n_layers']
        downstream_model = model_config['downstream_model']
        self.loss_type = model_config['loss_type']

        self.bert = BERT(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            max_length=max_length,
            position_embedding=position_embedding
        )
        if run_config['pretrained_path']:
            self.bert.load_state_dict(torch.load(run_config['pretrained_path'], map_location=self.device))
            #! be careful here, if the models don't have same names, it will throw an error
            #! here I use strict=False, so that unncessary weights for Pytorch-Lightning are not loaded
        
        self.model = BERTWithClosenessResNet(
            bert=self.bert,
            input_channels=num_attention_heads*num_hidden_layers,
            num_residual_blocks=downstream_n_layers,
            downstream_model=downstream_model,
            task=self.task,
            remove_backbone_width=remove_backbone_width
        )
        self.num_classes = self.model.num_classes

        if run_config['frozen']:
            print('Freezing the pretrained model.')
            self.freeze_pretrained_layers()

        self.val_data_dict = {'targets': [], 'logits': [], 'names': [], 'seq_lens': []}
        self.train_data_dict = {'targets': [], 'names': [], 'seq_lens': []}
        self.test_data_dict = {'targets': [], 'logits': [], 'names': [], 'seq_lens': []}


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
        input_ids, target, seq_len, name = batch
        logits = self.forward(input_ids)
        loss = self.loss(logits, target)
        if self.current_epoch == 0:
            self.train_data_dict['targets'].append(target)
            self.train_data_dict['names'].append(name[0].split('.')[0])
            self.train_data_dict['seq_lens'].append(seq_len.item())
        metrics = self.metrics(logits, target, mode='train')
        self.log("Training Loss", loss, sync_dist = True, prog_bar = True)
        self.log_dict(metrics, sync_dist = True)

        return loss
    
    def validation_step(
            self, 
            batch: typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str], 
            batch_idx: int
        ) -> torch.Tensor:
        input_ids, target, seq_len, name = batch
        logits = self.forward(input_ids)
        loss = self.loss(logits, target)
        metrics = self.metrics(logits, target, mode='val')
        if self.current_epoch % self.internal_log_frequency == 0:
            self.val_data_dict['targets'].append(target)
            self.val_data_dict['logits'].append(logits)
            self.val_data_dict['names'].append(name[0].split('.')[0])
            self.val_data_dict['seq_lens'].append(seq_len.item())
        self.log("Validation Loss", loss, sync_dist = True, prog_bar = True)
        self.log_dict(metrics, sync_dist = True)
        
        return loss

    def test_step(
            self, 
            batch: typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str], 
            batch_idx: int
        ) -> torch.Tensor:
        input_ids, target, seq_len, name = batch
        logits = self.forward(input_ids)
        loss = self.loss(logits, target)
        metrics = self.metrics(logits, target, mode='test')
        if self.current_epoch % self.internal_log_frequency == 0:
            self.test_data_dict['targets'].append(target)
            self.test_data_dict['logits'].append(logits)
            self.test_data_dict['names'].append(name[0].split('.')[0])
            self.test_data_dict['seq_lens'].append(seq_len.item())
        self.log("Test Loss", loss, sync_dist = True, prog_bar = True)
        self.log_dict(metrics, sync_dist = True)
        
        return loss

    def _get_alpha(self, target: torch.Tensor) -> torch.Tensor:
        classes, class_counts = target.unique(return_counts=True)
        if len(classes) < self.num_classes:
            classes, class_counts = self._fix_target_classes(classes, class_counts)
        class_freq = class_counts / class_counts.sum()
        alpha = 1.0/class_freq
        alpha = alpha / alpha.sum()
        return alpha
    

    def _fix_target_classes(self, classes, class_counts):
        full_range = torch.arange(0, self.num_classes, device=self.device)
        mask = torch.isin(full_range, classes)

        updated_tensor = full_range.clone()
        updated_tensor[mask] = classes

        new_indices = torch.where(~mask)[0]

        expanded_class_counts = torch.zeros_like(full_range, dtype=class_counts.dtype)
        expanded_class_counts[mask] = class_counts

        expanded_class_counts[new_indices] = 1e-10

        return updated_tensor, expanded_class_counts


    def loss(
            self, 
            logits: torch.Tensor, 
            target: torch.Tensor
        ) -> torch.Tensor:
        if self.class_weights_list is None:
            self.class_weights_tensor = None
        else:
            self.class_weights_tensor = torch.tensor(self.class_weights_list, device=self.device, dtype=torch.float32)

        if self.loss_type == 'focal_loss':
            if self.task == 'contact_map':
                self.loss_fn = FocalLossMultiClass(
                    alpha=self.alpha_focal_loss,
                    gamma=2,
                    ignore_index=-1,
                )
            elif self.task == 'distance_map':
                self.loss_fn = FocalLossMultiClass(
                    alpha=self._get_alpha(target),
                    gamma=2,
                    ignore_index=21,
                )
        else:
            raise ValueError('Loss function not supported.')
        # target [B,L,L], logits [B,C,L,L]
        loss = self.loss_fn(logits, target)
        return loss


    def metrics(
            self, 
            logits: torch.Tensor, 
            target: torch.Tensor,
            mode: typing.Optional[str] = None,
        ) -> dict:
        if self.task == 'contact_map':
            probs = torch.softmax(logits, dim=1)
            probs = probs[:, 1, :, :]
        elif self.task == 'distance_map':
            probs = self.model.cpc.compute_detailed_probabilities(logits)['contact_probability']
            target = self.model.dtcc.class_distances_to_binary(target)

        metrics = evaluate_prediction(
            predictions=probs.squeeze(),
            targets=target.squeeze(),
            mode=mode
        )
        return metrics
    
    def on_train_epoch_end(self) -> None:
        if self.current_epoch == 0:
            plots_num = int(np.ceil(np.sqrt(len(self.train_data_dict['targets']))))
            fig, ax = plt.subplots(plots_num, plots_num, figsize=(30, 30))
            for i, (target, name, seq_len) in enumerate(zip(self.train_data_dict['targets'], self.train_data_dict['names'], self.train_data_dict['seq_lens'])):
                ax[i//plots_num, i%plots_num].imshow(target.squeeze(0).cpu().numpy(), origin='lower', cmap='hot')
                ax[i//plots_num, i%plots_num].set_xlabel(f'{name} - {seq_len}')
                ax[i//plots_num, i%plots_num].set_ylabel(f'{name} - {seq_len}')
            self.logger.log_image(key = 'Targets Used For Training', images = [fig])
            plt.tight_layout()
            plt.close(fig)
            self.train_data_dict = {'targets': [], 'names': [], 'seq_lens': []}
        

    def on_validation_epoch_end(self) -> None:
        if self.current_epoch % self.internal_log_frequency == 0:
            plots_num = int(np.ceil(np.sqrt(len(self.val_data_dict['targets']))))
            fig, ax = plt.subplots(plots_num, plots_num, figsize=(12, 10))
            ind_fig, ind_ax = plt.subplots(1,1)
            for i, (target, logits, name, seq_len) in enumerate(zip(self.val_data_dict['targets'], self.val_data_dict['logits'], self.val_data_dict['names'], self.val_data_dict['seq_lens'])):
                if self.task == 'contact_map':
                    probs = torch.softmax(logits, dim=1)
                    probs = probs[:, 1, :, :]
                elif self.task == 'distance_map':
                    probs = self.model.cpc.compute_detailed_probabilities(logits)['contact_probability']
                    target = self.model.dtcc.class_distances_to_binary(target)
                plot_contacts_and_predictions(
                    probs.squeeze().cpu().numpy(),
                    target.squeeze().cpu().numpy(),
                    ax=ax[i//plots_num, i%plots_num]
                )
                plot_contacts_and_predictions(
                    probs.squeeze().cpu().numpy(),
                    target.squeeze().cpu().numpy(),
                    ax=ind_ax,
                    ms=5
                )
                ax[i//plots_num, i%plots_num].set_xlabel(f'{name} - {seq_len}')
                ax[i//plots_num, i%plots_num].set_ylabel(f'{name} - {seq_len}')
                ind_ax.set_xlabel(f'{name} - {seq_len}')
                ind_ax.set_ylabel(f'{name} - {seq_len}')
                ind_fig.tight_layout()
                path = f"{self.logger.save_dir}/plots/{self.logger.experiment.project}/{self.logger.experiment.name}/{self.current_epoch}"
                os.makedirs(path, exist_ok=True)
                ind_fig.savefig(f"{path}/{name}.png")
                ind_ax.clear()

            self.logger.log_image(key = 'Outputs', images = [fig])
            fig.tight_layout()
            plt.close(fig)
            self.val_data_dict = {'targets': [], 'logits': [], 'names': [], 'seq_lens': []}
            
        
    def on_test_epoch_end(self) -> None:
        ind_fig, ind_ax = plt.subplots(1,1, figsize=(12, 10))
        for i, (target, logits, name, seq_len) in enumerate(zip(self.test_data_dict['targets'], self.test_data_dict['logits'], self.test_data_dict['names'], self.test_data_dict['seq_lens'])):
            if self.task == 'contact_map':
                probs = torch.softmax(logits, dim=1)
                probs = probs[:, 1, :, :]
            elif self.task == 'distance_map':
                probs = self.cpc.compute_detailed_probabilities(logits)['contact_probability']
                target = self.dtcc.class_distances_to_binary(target)
            colors = [
                jc.custom_colors['light_grey'],      # Light (low values)
                jc.custom_colors['julich_blue_2'],   # Medium
                jc.custom_colors['julich_blue_1'],   # Dark (high values)
            ]
            custom_cmap = LinearSegmentedColormap.from_list('julich', colors, N=100)
            plot_contacts_and_predictions(
                probs.squeeze().cpu().numpy(),
                target.squeeze().cpu().numpy(),
                ax=ind_ax,
                ms=10,
                cmap=custom_cmap,
            )
            ind_ax.set_xlabel(f'{name} - {seq_len}')
            ind_ax.set_ylabel(f'{name} - {seq_len}')
            ind_fig.tight_layout()
            path = f"{self.logger.save_dir}/plots/{self.logger.experiment.project}/{self.logger.experiment.name}/{self.current_epoch}"
            os.makedirs(path, exist_ok=True)
            ind_fig.savefig(f"{path}/{name}.svg", bbox_inches='tight', transparent=True, dpi=300, format='svg')
            ind_ax.clear()

            self.logger.log_image(key = 'Outputs', images = [ind_fig])
        self.test_data_dict = {'targets': [], 'logits': [], 'names': [], 'seq_lens': []}

    @staticmethod
    def plot_blind_contacts(
        predictions: Union[torch.Tensor, np.ndarray],
        ax: Optional[mpl.axes.Axes] = None,
        cmap: str = "Blues",
        ms: float = 1,
        contact_threshold: str = "top-L",  # Can be "top-L", "top-L/2", "top-L/5" or a float between 0 and 1
        title: bool = True
    ) -> None:
        """
        Visualize contact predictions without ground truth contacts.
        
        Args:
            predictions: Contact prediction matrix (can be torch.Tensor or numpy.ndarray)
            ax: Matplotlib axis for plotting (optional)
            cmap: Colormap for the background probabilities
            ms: Marker size for predicted contacts
            contact_threshold: How to determine contacts - either "top-L", "top-L/2", "top-L/5" or a float threshold
            title: Whether to show title with prediction info
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        
        if predictions.ndim == 3:
            predictions = predictions[0]  # Take first batch if batched
            
        seqlen = predictions.shape[0]
        
        # Create mask for lower triangle and short-range contacts
        relative_distance = np.add.outer(-np.arange(seqlen), np.arange(seqlen))
        bottom_mask = relative_distance < 0
        invalid_mask = np.abs(np.add.outer(np.arange(seqlen), -np.arange(seqlen))) < 6
        
        # Mask invalid regions
        predictions = predictions.astype(np.float32)
        predictions[invalid_mask] = -np.inf
        masked_image = np.ma.masked_where(bottom_mask, predictions)
        
        # Determine contacts based on threshold
        if contact_threshold == "top-L":
            top_k = seqlen
        elif contact_threshold == "top-L/2":
            top_k = seqlen // 2
        elif contact_threshold == "top-L/5":
            top_k = seqlen // 5
        else:
            try:
                threshold = float(contact_threshold)
                if not 0 <= threshold <= 1:
                    raise ValueError
                pred_contacts = predictions >= threshold
                top_k = None
            except ValueError:
                raise ValueError("contact_threshold must be 'top-L', 'top-L/2', 'top-L/5' or a float between 0 and 1")
        
        if top_k is not None:
            topl_val = np.sort(predictions.reshape(-1))[-top_k]
            pred_contacts = predictions >= topl_val
        
        # Create plot
        if ax is None:
            ax = plt.gca()
        
        # Plot probability background
        img = ax.imshow(masked_image, cmap=cmap)
        plt.colorbar(img, ax=ax, label='Contact Probability')
        
        # Plot predicted contacts
        contacts = ax.plot(*np.where(pred_contacts & ~bottom_mask), 'o', 
                        c='blue', ms=ms, label='Predicted Contacts')[0]
        
        if title:
            if top_k is not None:
                ax.set_title(f'Predicted Contacts (top {top_k})')
            else:
                ax.set_title(f'Predicted Contacts (threshold={contact_threshold})')
        
        ax.axis("square")
        ax.set_xlim([0, seqlen])
        ax.set_ylim([0, seqlen])
        ax.legend(loc="best")
            

    
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        if self.use_scheduler:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=0.1 * self.max_epochs,
                num_training_steps=self.max_epochs
            )
            return [self.optimizer], [self.scheduler]
        else:
            return [self.optimizer]
    
    def freeze_pretrained_layers(self) -> None:
        for param in self.bert.parameters():
            param.requires_grad = False

