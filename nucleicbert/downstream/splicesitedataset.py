import os
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset
from nucleicbert.pretrain.utils import pad

class RNASpliceSiteDataset(Dataset):

    def __init__(
            self,
            input_dir: os.PathLike,
            tokenizer,
            max_length: int = 1024,
            min_length: int = 10,
            mode: str = 'Train',
            task: str = 'acceptor',
            species: str = None,

        ) -> None:
        super(RNASpliceSiteDataset, self).__init__()
        self.max_length = max_length
        self.min_length = min_length
        self.tokenizer = tokenizer
        self.mode = mode
        self.species = species
        self.task = task
        
        assert max_length > 0, "max_length should be greater than 0"
        assert min_length >= 0, "min_length should be greater than or equal to 0"

        if mode == 'Train' or mode == 'train':
            self.mode = 'Train'
            self.df = self._load_csv_data(input_dir)
        elif mode == 'Val' or mode == 'val':
            self.mode = 'Val'
            self.df = self._load_csv_data(input_dir)
        elif mode == 'test' or mode == 'Test':
            self.mode = 'Test'
            self.df = self._load_input_data(input_dir)
        else:
            raise ValueError("mode should be either 'train' or 'test'")
        

    
    def _load_csv_data(self, input_dir):
        """Load sequences and structures from CSV files in the input directory."""
        csv_files = glob.glob(f"{input_dir}/{self.mode}_{self.task}_*.csv")
        print(f"Using following files for {self.task} task: {csv_files}")
        dataframes = []
        for csv_file in csv_files:
            # csv_file_path = os.path.join(input_dir, csv_file)
            df = self._load_csv_sequences(csv_file)
            dataframes.append(df)
        
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        return combined_df
    
    def _load_input_data(self, input_dir):
        """Load sequences from files in the input directory."""
        species_dir = os.path.join(input_dir, self.species)
        input_files = [os.path.join(species_dir, f'SA_sequences_{self.task}_400_Final_3.fasta'), os.path.join(species_dir, f'SA_sequences_{self.task}_400_Final_3.fasta')]
        dataframes = []
        
        for input_file in input_files:
            df = self._load_input_lines(input_file)
            dataframes.append(df)
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        return combined_df
    
    def _load_csv_sequences(self, csv_file_path):
        """Load sequences and structures from a CSV file."""
        try:
            df = pd.read_csv(csv_file_path, sep = ';', header=None)
            df['label'] = df.iloc[:, 2]
            df['sequence'] = df.iloc[:, 1]
            return df
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {e}")
        
    def _load_input_lines(self, filename):
        """Load sequences from files in a directory."""
        sequences = []
        with open(filename, 'r') as f:
            for line in f:
                sequence = line.strip()
                sequences.append(sequence)
        with open(filename.replace('sequences', 'labels'), 'r') as f:
            labels = []
            for line in f:
                labels.append(int(line.strip()))
        df = pd.DataFrame({'sequence': sequences, 'label': labels})
        return df

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
        sequence = row['sequence']
        label = row['label']
        
        input_ids, label, seq_len = self.preprocess(sequence, label)
        
        return input_ids, label, sequence

if __name__ == '__main__':
    import os
    from transformers import PreTrainedTokenizerFast
    from torch.utils.data import DataLoader

    # Example usage
    input_dir = '../data/splice_data/Benchmarks/'
    tokenizer = PreTrainedTokenizerFast(tokenizer_file='nucleicbert/tokenizers/noncoding_seqs.json')
    
    dataset = RNASpliceSiteDataset(
        input_dir=input_dir,
        tokenizer=tokenizer,
        max_length=400,
        min_length=10,
        mode='test',
        task = 'donor',
        species='Danio'
    )
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    labels_sum = 0
    i = 0
    for batch in dataloader:
        input_ids, labels, sequences = batch
        print(f"Input IDs: {input_ids}, Labels: {labels}")
        i+= 1
        if i > 5:
            break
