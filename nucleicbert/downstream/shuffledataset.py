import os
import typing
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from nucleicbert.pretrain.utils import pad

class RNAShuffleDetectionDataset(Dataset):
    """
    Dataset class for RNA Sequence Shuffling Detection.
    
    This dataset creates pairs of real RNA sequences and their shuffled counterparts
    for training a model to distinguish between real and artificially shuffled sequences.

    Args:
        input_seq: List of RNA sequences or path to directory containing sequence files
        tokenizer: Tokenizer for converting sequences to token IDs
        shuffle_method: Method for shuffling ('complete', 'dinucleotide', 'kmer', 'pair_swap')
        kmer_size: Size of k-mers to preserve when using 'kmer' shuffle method
        max_length: Maximum sequence length
        min_length: Minimum sequence length
        augmentation_factor: How many shuffled versions to create per real sequence
        enable_shuffling: Whether to generate shuffled sequences or not
        shuffle_percentage: Percentage of sequence to shuffle (0.0-1.0)
    """
    def __init__(
            self,
            input_seq: typing.Union[typing.List[str], os.PathLike],
            tokenizer,
            shuffle_method: str = 'complete',
            kmer_size: int = 2,
            max_length: int = 1024,
            min_length: int = 10,
            augmentation_factor: int = 1,
            enable_shuffling: bool = True,
            skip_real: bool = False,
            shuffle_percentage: float = 1.0
        ) -> None:
        super(RNAShuffleDetectionDataset, self).__init__()
        
        assert shuffle_method in ['complete', 'dinucleotide', 'kmer', 'pair_swap'], \
            f"Shuffle method {shuffle_method} is not supported. Choose 'complete', 'dinucleotide', 'kmer', or 'pair_swap'"
        assert kmer_size > 0, "kmer_size should be greater than 0"
        assert max_length > 0, "max_length should be greater than 0"
        assert min_length > 0, "min_length should be greater than 0"
        assert augmentation_factor > 0, "augmentation_factor should be greater than 0"
        assert 0.0 <= shuffle_percentage <= 1.0, "shuffle_percentage should be between 0.0 and 1.0"

        self.skip_real = skip_real
        
        if isinstance(input_seq, list):
            self.input_lines = input_seq
            self.individual_data_file_names = [f"sequence_{i}" for i in range(len(self.input_lines))]
        elif isinstance(input_seq, str):
            if os.path.isdir(input_seq):
                self.individual_data_file_names = sorted([os.path.basename(file) for file in os.listdir(input_seq)])
                self.input_lines = self._load_input_lines(input_seq)
            elif input_seq.endswith('.csv'):
                self.input_lines, self.individual_data_file_names, self.structures = self._load_csv_sequences(input_seq)
            else:
                with open(input_seq, 'r') as f:
                    self.input_lines = [line.strip() for line in f]
                self.individual_data_file_names = [f"sequence_{i}" for i in range(len(self.input_lines))]
        else:
            raise ValueError("input_seq should be either a list of strings or a path to a directory/file")
        
        self.max_length = max_length
        self.min_length = min_length
        self.tokenizer = tokenizer
        self.shuffle_method = shuffle_method
        self.kmer_size = kmer_size
        self.augmentation_factor = augmentation_factor
        self.enable_shuffling = enable_shuffling
        self.shuffle_percentage = shuffle_percentage
        
        self._prepare_data()
    
    def _load_csv_sequences(self, csv_file_path):
        """Load sequences and structures from a CSV file."""
        try:
            df = pd.read_csv(csv_file_path)
            if 'Sequence' not in df.columns:
                raise ValueError("CSV file must contain a 'Sequence' column")
            
            sequences = df['Sequence'].tolist()
            

            if 'Name' in df.columns:
                names = df['Name'].tolist()
            else:
                names = [f"csv_sequence_{i}" for i in range(len(sequences))]
            

            structures = []
            if 'Structure' in df.columns:
                structures = df['Structure'].tolist()
            
            return sequences, names, structures
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {e}")
        
    def _load_input_lines(self, directory_path):
        """Load sequences from files in a directory."""
        sequences = []
        for filename in self.individual_data_file_names:
            filepath = os.path.join(directory_path, filename)
            with open(filepath, 'r') as f:

                sequence = ''.join(line.strip() for line in f if not line.startswith('>'))
                sequences.append(sequence)
        return sequences
        
    def _prepare_data(self):
        """Process and filter sequences, then create dataset with real and shuffled pairs."""
        data_dict = {'sequence': [], 'is_real': [], 'original_idx': [], 'name': []}
        

        for i, seq in enumerate(self.input_lines):
            name_info = self.individual_data_file_names[i] if hasattr(self, 'individual_data_file_names') else f"sequence_{i}"
            

            if len(seq) < self.min_length or len(seq) > self.max_length:
                continue
                

            if not self.skip_real:
                data_dict['sequence'].append(seq)
                data_dict['is_real'].append(1)  # 1 for real
                data_dict['original_idx'].append(i)
                data_dict['name'].append(name_info)
            

            if self.enable_shuffling and self.augmentation_factor > 0:
                for j in range(self.augmentation_factor):
                    if self.shuffle_method == 'pair_swap' and hasattr(self, 'structures') and len(self.structures) > i:
                        structure = self.structures[i]
                        shuffled_seq = self._shuffle_paired_residues(seq, structure)
                    else:
                        shuffled_seq = self._shuffle_sequence(seq)
                    data_dict['sequence'].append(shuffled_seq)
                    data_dict['is_real'].append(0)  # 0 for shuffled
                    data_dict['original_idx'].append(i)
                    data_dict['name'].append(f"{name_info}_shuffled_{j}")
        
        self.data_df = pd.DataFrame(data_dict)
        print(f"Created dataset with {len(self.data_df)} entries: {len(self.data_df[self.data_df['is_real'] == 1])} real, "
              f"{len(self.data_df[self.data_df['is_real'] == 0])} shuffled")
    
    def _find_pairs_from_structure(self, structure):
        """
        Find paired positions in an RNA secondary structure.
        
        Args:
            structure: RNA secondary structure in WUSS notation
            
        Returns:
            list of tuples: Each tuple contains indices of paired nucleotides
        """

        opening_brackets = "([{<"
        closing_brackets = ")]}>"
        

        bracket_pairs = {')': '(', ']': '[', '}': '{', '>': '<'}
        reverse_bracket_pairs = {'(': ')', '[': ']', '{': '}', '<': '>'}
        

        bracket_stacks = {'(': [], '[': [], '{': [], '<': []}
        pairs = []
        
        for i, char in enumerate(structure):
            if char in opening_brackets:

                bracket_stacks[char].append(i)
            elif char in closing_brackets:

                opening_char = bracket_pairs[char]
                stack = bracket_stacks[opening_char]
                
                if not stack:
                    continue  # Unmatched closing bracket, skip
                

                opening_idx = stack.pop()
                

                pairs.append((opening_idx, i))
        
        return pairs
    
    def _shuffle_paired_residues(self, sequence, structure):
        """
        Shuffle a sequence by swapping paired residues according to the secondary structure.
        
        Args:
            sequence: RNA sequence
            structure: RNA secondary structure in WUSS notation
            
        Returns:
            str: Shuffled sequence where paired residues are swapped
        """

        if self.shuffle_percentage <= 0:
            return sequence
        assert len(sequence) == len(structure), "Sequence and structure must be of the same length"
        

        paired_positions = self._find_pairs_from_structure(structure)
        
        if not paired_positions:

            return self._shuffle_sequence(sequence)
        

        if self.shuffle_percentage < 1.0:

            num_pairs_to_shuffle = int(len(paired_positions) * self.shuffle_percentage)
            if num_pairs_to_shuffle == 0 and paired_positions:
                num_pairs_to_shuffle = 1  # Shuffle at least one pair if any exist
                

            pairs_to_shuffle = random.sample(paired_positions, num_pairs_to_shuffle)
        else:

            pairs_to_shuffle = paired_positions
        

        result = list(sequence)
        
        for pos1, pos2 in pairs_to_shuffle:

            result[pos1], result[pos2] = result[pos2], result[pos1]
        
        return ''.join(result)
    
    def _shuffle_sequence(self, sequence):
        """
        Shuffle a sequence according to the specified method.
        
        The shuffling is controlled by shuffle_percentage, which determines
        what portion of the sequence will be shuffled.
        """

        if self.shuffle_percentage <= 0:
            return sequence
            

        if self.shuffle_percentage < 1.0:
            # Determine which positions to shuffle
            seq_length = len(sequence)
            num_positions_to_shuffle = int(seq_length * self.shuffle_percentage)
            positions_to_shuffle = sorted(random.sample(range(seq_length), num_positions_to_shuffle))
            

            chars_to_shuffle = [sequence[i] for i in positions_to_shuffle]
            

            shuffled_chars = self._apply_shuffle_method(chars_to_shuffle)
            

            result = list(sequence)
            for i, pos in enumerate(positions_to_shuffle):
                result[pos] = shuffled_chars[i]
                
            return ''.join(result)
        

        return self._apply_shuffle_method(list(sequence))
    
    def _apply_shuffle_method(self, chars):
        """Apply the selected shuffle method to a list of characters."""
        if self.shuffle_method == 'complete':
            random.shuffle(chars)
            return ''.join(chars)
        
        elif self.shuffle_method == 'dinucleotide':
            sequence = ''.join(chars)
            return self._dinucleotide_shuffle(sequence)
            
        elif self.shuffle_method == 'kmer':
            sequence = ''.join(chars)
            return self._kmer_shuffle(sequence)
            
        else:
            random.shuffle(chars)
            return ''.join(chars)
    
    def _dinucleotide_shuffle(self, sequence):
        """Dinucleotide-preserving shuffling (Altschul-Erikson algorithm)."""
        if len(sequence) <= 2:
            return sequence  
            

        seq_list = list(sequence)
        

        dinucleotides = [(seq_list[i], seq_list[i+1]) for i in range(len(seq_list)-1)]
        

        shuffled = [seq_list[0]]
        current = seq_list[0]
        
        while dinucleotides:

            options = [i for i, (first, _) in enumerate(dinucleotides) if first == current]
            
            if not options:
                break  # No valid options, might happen in edge cases
                
            choice = random.choice(options)
            _, next_char = dinucleotides.pop(choice)
            

            shuffled.append(next_char)
            current = next_char
            
        # If we didn't use all dinucleotides, it's usually due to an isolated cycle
        # In that case, just append the remaining shuffled dinucleotides
        if len(shuffled) < len(sequence):
            remaining = []
            for first, second in dinucleotides:
                remaining.extend([first, second])
            random.shuffle(remaining)
            shuffled.extend(remaining)
            
        return ''.join(shuffled[:len(sequence)])  
        
    def _kmer_shuffle(self, sequence):
        """k-mer preserving shuffling."""
        if len(sequence) <= self.kmer_size:
            return sequence  # Too short to shuffle meaningfully
            
        kmers = [sequence[i:i+self.kmer_size] 
                 for i in range(len(sequence) - self.kmer_size + 1)]
        
        random.shuffle(kmers)
        
        result = kmers[0]
        for kmer in kmers[1:]:
            result += kmer[-1]
            
        if len(result) < len(sequence):
            result += sequence[-(len(sequence)-len(result)):]
        elif len(result) > len(sequence):
            result = result[:len(sequence)]
            
        return result

    def preprocess(self, sequence, is_real):
        """
        Preprocess a sequence for model input.
        
        Args:
            sequence: RNA sequence
            is_real: Whether this is a real (1) or shuffled (0) sequence
            
        Returns:
            input_ids: Tokenized sequence
            label: Binary label (real=1, shuffled=0)
        """
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
        

        label = torch.tensor(is_real, dtype=torch.float)
        
        return input_ids, label, len(sequence)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        row = self.data_df.iloc[idx]
        sequence = row['sequence']
        is_real = row['is_real']
        
        input_ids, label, seq_len = self.preprocess(sequence, is_real)
        
        return input_ids, label, sequence, row['name']



if __name__ == "__main__":

    class DummyTokenizer:
        def __init__(self):
            self.vocab = {'[CLS]': 0, '[SEP]': 1, 'A': 2, 'C': 3, 'G': 4, 'T': 5, 'N': 6}
            
        def tokenize(self, text):
            return list(text)
            
        def convert_tokens_to_ids(self, tokens):
            return [self.vocab.get(t, self.vocab['N']) for t in tokens]
    

    sequences = [
        "CAACGUUCACUUUGCUGAUACGCAAAGCGCAUCACACCUCAGGCCAUGGAACGGGGACCUGGG",
    ]
    
    structures = [
        "...[[[[[.(((((........)))))..........((((((((...]]]]].)).))))))",  # Example structure
    ]
    

    tokenizer = DummyTokenizer()
    dataset = RNAShuffleDetectionDataset(
        sequences,
        tokenizer,
        shuffle_method='complete',
        augmentation_factor=1
    )
    

    combined_data = pd.DataFrame({
        'Name': [f'seq_{i}' for i in range(len(sequences))],
        'Sequence': sequences,
        'Structure': structures
    })
    combined_data.to_csv('temp_combined_data.csv', index=False)
    
    pair_swap_dataset = RNAShuffleDetectionDataset(
        'temp_combined_data.csv',
        tokenizer,
        shuffle_method='pair_swap',
        augmentation_factor=1
    )
    

    for i in range(len(pair_swap_dataset)):
        input_ids, label, seq, name = pair_swap_dataset[i]
        print(f"Sample {i}: {seq}, Label: {label.item()}")