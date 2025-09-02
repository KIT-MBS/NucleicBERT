import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os

from nucleicbert.pretrain.utils import pad


def dot_bracket_to_pair_matrix(db_notation: str) -> np.ndarray:
    """Convert dot-bracket notation to 2D pairing matrix."""
    seq_len = len(db_notation)
    pair_mat = np.zeros((seq_len, seq_len), dtype=np.float32)
    
    stacks = {
        "(": [], "[": [], "{": [], "<": [],
    }
    closing = {"(": ")", "[": "]", "{": "}", "<": ">"}
    opening = {v: k for k, v in closing.items()}
    
    for i in range(seq_len):
        char = db_notation[i]
        
        if char in stacks:  # Opening bracket
            stacks[char].append(i)
        elif char in opening:  # Closing bracket
            open_char = opening[char]
            if stacks[open_char]:
                j = stacks[open_char].pop()
                pair_mat[i, j] = 1.0
                pair_mat[j, i] = 1.0
            
    return pair_mat


def load_data(data_dir):
    """Load the data from the data directory."""
    if os.sep not in data_dir:
        raise ValueError("Please provide the full path to the data directory.")
    
    data = pd.read_csv(data_dir)
    return data


class RNASecStr2DDataset(Dataset):
    """Dataset for 2D secondary structure prediction."""
    
    def __init__(
        self,
        data_dir,
        tokenizer,
        max_length,
        min_length,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length
        
        self.data = load_data(data_dir)
    
    def preprocess(self, sequence):
        """Preprocess the sequence."""
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
        
        actual_length = len(input_ids)
        input_ids = pad(input_ids, self.max_length)
        
        return input_ids, actual_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = self.data['Sequence'].iloc[idx]
        structure = self.data['Structure'].iloc[idx]
        

        structure = structure[:self.max_length-2]
        

        input_ids, actual_length = self.preprocess(sequence)
        

        pair_matrix = dot_bracket_to_pair_matrix(structure)
        pair_matrix = torch.tensor(pair_matrix, dtype=torch.float32)
        
        # Pad the pair matrix to match max_length (accounting for [CLS] and [SEP])
        # The matrix represents only the actual sequence (without special tokens)
        padded_size = self.max_length - 2  # Size without [CLS] and [SEP]
        if pair_matrix.shape[0] < padded_size:
            padding = padded_size - pair_matrix.shape[0]
            pair_matrix = F.pad(pair_matrix, (0, padding, 0, padding), value=0)
        
        return input_ids, pair_matrix, sequence[:self.max_length-2], structure

