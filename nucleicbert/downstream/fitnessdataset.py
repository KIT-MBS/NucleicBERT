import os
import pandas as pd
import torch
from torch.utils.data import Dataset

from nucleicbert.pretrain.utils import pad

class FitnessDataset(Dataset):

    def __init__(
            self,
            input_dir: os.PathLike,
            tokenizer,
            max_length: int = 1024,
            min_length: int = 10,
        ) -> None:
        super(FitnessDataset, self).__init__()
        self.max_length = max_length
        self.min_length = min_length
        self.tokenizer = tokenizer
        
        assert max_length > 0, "max_length should be greater than 0"
        assert min_length > 0, "min_length should be greater than 0"
        self.df = self._load_csv_data(input_dir)
        

    
    def _load_csv_data(self, csv_file_path):
        """Load sequences and structures from a CSV file."""
        try:
            df = pd.read_csv(csv_file_path)
            return df
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {e}")

    def preprocess(self, sequence, label):

        sequence = sequence.upper()
        sequence = sequence.replace('U', 'T')  
        sequence = sequence.replace('X', 'N')  
        sequence = sequence.replace('I', 'N')  
        sequence = sequence[:self.max_length-2]
        

        input_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sequence))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        

        input_ids = torch.cat([
            torch.tensor([self.tokenizer.vocab['[CLS]']]), 
            input_ids, 
            torch.tensor([self.tokenizer.vocab['[SEP]']])
        ], dim=0)
        input_ids = pad(input_ids, self.max_length)
        

        label = torch.tensor(label, dtype=torch.float)
        
        return input_ids, label, len(sequence)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        row = self.df.iloc[idx]
        sequence = row['sequences']
        fitness = row['fit_mean']
        num_mutations = row['number_mutations']
        
        input_ids, fitness, seq_len = self.preprocess(sequence, fitness)
        
        return input_ids, fitness, sequence, num_mutations

if __name__ == '__main__':
    import os
    from transformers import PreTrainedTokenizerFast
    from torch.utils.data import DataLoader

    # Example usage
    input_dir = '../data/mutation_data/train/train_data.csv'
    tokenizer = PreTrainedTokenizerFast(tokenizer_file='nucleicbert/tokenizers/noncoding_seqs.json')
    
    dataset = FitnessDataset(
        input_dir=input_dir,
        tokenizer=tokenizer,
        max_length=400,
        min_length=10,
    )
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    labels_sum = 0
    i = 0
    for batch in dataloader:
        input_ids, labels, sequences, num_mutations = batch
        print(f"Input IDs: {input_ids}, Labels: {labels}")
        i+= 1
        if i > 5:
            break
