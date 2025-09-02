import torch
import pytorch_lightning as pl
import torchmetrics as tm
import pandas as pd
import tqdm
import os


from nucleicbert.models.bert import BERT, NB_CONFIG
from nucleicbert.pretrain.utils import Config
from nucleicbert.pretrain.pretrainingdataset import RNASeqDataset 
SAVE_DIR = 'results/per_seq_analysis/val_data/'

class ModelInference:
    def __init__(
            self, 
            model_path, 
            device='cuda',
            tokenizer=None, 
            model_config=None,
        ):
        """
        Initialize the inference class
        
        Args:
            model_path (str): Path to the saved model checkpoint
            tokenizer (optional): Tokenizer to use
            model_config (dict, optional): Model configuration
        """
        self.device = device
        self.model_path = model_path
        if model_config is None:
            model_config = NB_CONFIG
        self.model_config = model_config
        
        self.tokenizer = tokenizer
        
        self.model = self._load_model(model_path)
        
        self.accuracy_fn = tm.classification.MulticlassAccuracy(
            num_classes=self.model_config.get('vocab_size'),
            ignore_index=-100,
            multidim_average='samplewise',
        ).to(self.device)
        self.loss_fn = torch.nn.CrossEntropyLoss(
            ignore_index=-100,
            reduction='none',
        ).to(self.device)
        
        os.makedirs(SAVE_DIR, exist_ok=True)
    
    def _load_model(self, model_path):
        """
        Load the pre-trained model from a checkpoint
        
        Args:
            model_path (str): Path to the model checkpoint
        
        Returns:
            PreTrainingModule: Loaded model
        """

        model = BERT(**self.model_config)
        if self.model_path is not None:
            model.load_state_dict(torch.load(model_path))
        model = model.to(self.device)
        model.eval()  # Set to evaluation mode
        return model
    
    def evaluate_dataset(self, dataloader, save_every=10, checkpoint_file=f'{SAVE_DIR}/accuracy_checkpoint_val_data_notpretrained.csv'):
        """
        Evaluate the model on a given dataset with continuous saving
        
        Args:
            dataloader: Data loader for the validation dataset
            save_every (int): Save checkpoint after this many batches
            checkpoint_file (str): Path to save checkpoint file
        
        Returns:
            dict: Evaluation metrics
        """
        per_seq_acc_dict = {
            'accuracy': [],
            'ppl': [],
            'sequence': [],
            'input_tokens': [],
            'target_tokens': [],
        }
        
        start_batch = 0
        if os.path.exists(checkpoint_file):
            checkpoint_df = pd.read_csv(checkpoint_file)
            per_seq_acc_dict['accuracy'] = checkpoint_df['accuracy'].tolist()
            per_seq_acc_dict['ppl'] = checkpoint_df['ppl'].tolist()
            per_seq_acc_dict['sequence'] = checkpoint_df['sequence'].tolist()
            per_seq_acc_dict['input_tokens'] = checkpoint_df['input_tokens'].tolist()
            per_seq_acc_dict['target_tokens'] = checkpoint_df['target_tokens'].tolist()
            
            start_batch = len(checkpoint_df) // dataloader.batch_size
            print(f"Resuming from batch {start_batch} with {len(checkpoint_df)} samples already processed")
        
        batch_count = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm.tqdm(dataloader)):
                if batch_idx < start_batch:
                    continue
                    
                input_ids, target_ids, seq = batch[0], batch[1], batch[-1]
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                logits = self.model(input_ids)
                
                _, topk_idx = torch.topk(torch.softmax(logits, dim=-1), dim=-1, k=1)
                preds = topk_idx.squeeze(dim=-1)
                
                acc = self.accuracy_fn(preds, target_ids)

                loss = self.loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                loss_per_sample = loss.view(target_ids.size(0), -1).mean(dim=-1)
                ppl = torch.exp(loss_per_sample)
                
                target_ids[target_ids == -100] = 1
                input_tokens = list(map(self.tokenizer.convert_ids_to_tokens, input_ids.squeeze().tolist()))
                target_tokens = list(map(self.tokenizer.convert_ids_to_tokens, target_ids.squeeze().tolist()))

                per_seq_acc_dict['ppl'].extend(ppl.tolist())
                per_seq_acc_dict['accuracy'].extend(acc.tolist())
                per_seq_acc_dict['sequence'].extend(seq)
                per_seq_acc_dict['input_tokens'].extend(input_tokens)
                per_seq_acc_dict['target_tokens'].extend(target_tokens)
                
                batch_count += 1
                
                if batch_count % save_every == 0:
                    temp_df = pd.DataFrame(per_seq_acc_dict)
                    temp_df.to_csv(checkpoint_file, index=False)
                    print(f"Saved checkpoint after batch {batch_idx+1}/{len(dataloader)}")
        
        final_df = pd.DataFrame(per_seq_acc_dict)
        final_df.to_csv(f'{SAVE_DIR}/accuracy_val_data_notpretrained.csv', index=False)
        
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            print("Evaluation completed successfully, removed checkpoint file")
            
        return per_seq_acc_dict
    
    def generate_sequence(self, masked_sequence, max_iterations=10):
        """
        Generate a complete sequence from a masked sequence
        
        Args:
            masked_sequence (torch.Tensor): Input sequence with masked tokens
            max_iterations (int): Maximum number of iterations for generation
        
        Returns:
            torch.Tensor: Completed sequence
        """
        self.model.eval()
        sequence = masked_sequence
        input_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sequence))
        input_ids = torch.tensor(input_ids).to(self.device).unsqueeze(0)
        
        
        with torch.no_grad():
            for _ in range(max_iterations):
                logits = self.model(input_ids)
                
                masked_positions = (input_ids == self.tokenizer.vocab['[MASK]'])
                
                if not masked_positions.any():
                    break
                
                masked_logits = logits[masked_positions]
                predicted_tokens = torch.argmax(masked_logits, dim=-1)
                
                input_ids[masked_positions] = predicted_tokens
        sequence = self.tokenizer.decode(input_ids.squeeze().tolist()).replace(' ', '').upper()
        # sequence = [''.join(res) for res in sequence]

        return sequence

# Example usage
if __name__ == '__main__':
    from transformers import PreTrainedTokenizerFast
    pl.seed_everything(42, workers=True)
    # Configuration
    # model_path = 'weights/pretrained.pt'
    model_path = None
    
    # Initialize tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file='nucleicbert/tokenizers/noncoding_seqs.json')
    
    
    # Initialize inference
    inference = ModelInference(
        model_path,
        tokenizer=tokenizer,
    )
    data_path = "../mars_data/processed_data_python/validation_rna_sequences.csv"
    data = pd.read_csv(data_path)
    sequences = data['sequence'].tolist()
    labels = data['type'].tolist()
    val_dataset = RNASeqDataset(
        input=sequences,
        tokenizer=tokenizer,
        mask_lm_prob=0.15,
        max_length=1024,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=True,
    )
    # Use the modified evaluate_dataset method with checkpoint capability
    results = inference.evaluate_dataset(val_dataloader, save_every=5)

    
    # Example of sequence generation
    # Assume you have a masked sequence
    # masked_sequence = 'AUGCUAGCUAGCAUUG[MASK]GGUGUG'
    # completed_sequence = inference.generate_sequence(masked_sequence)
    # print("Completed Sequence:", completed_sequence)