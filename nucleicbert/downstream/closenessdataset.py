import os
import typing
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from nucleicbert.pretrain.utils import load_input_lines, load_targets, bin_distances_2d, remove_backbone_contacts

class RNAClosenessDataset(Dataset):
    """
    Dataset class for RNA Contact Maps. 


    Args:
        config: The config dictionary
        maskmaker: The MaskMaker object
        mode: The mode of the dataset. Can be either 'pretrain' or 'downstream'

    """
    def __init__(
            self,
            input_seq: typing.Union[typing.List[str], os.PathLike],
            target: typing.Union[typing.List[np.ndarray], os.PathLike],
            tokenizer,
            task: str = 'contact_map',
            remove_backbone_width: int = 0,
            distance_threshold: float = 8.0,
            max_length: int = 512,
            min_length: int =  1,
        ) -> None:
        super(RNAClosenessDataset, self).__init__()
        assert task in ['contact_map', 'distance_map'], f"Task {task} is not supported. Please choose either 'contact_map' or 'distance_map'"
        assert remove_backbone_width >= 0, "remove_backbone_width should be greater than or equal to 0"
        assert distance_threshold > 0, "distance_threshold should be greater than 0"
        if type(input_seq) is list and type(target) is list:
            self.input_lines = input_seq
            self.targets = target
        elif type(input_seq) is str and type(target) is str:
            self.individual_data_file_names = sorted([os.path.basename(file) for file in os.listdir(input_seq)])
            self.input_lines = load_input_lines(input_seq)
            self.targets = load_targets(target)
        else:
            raise ValueError("Input and target should be either list of strings/np.ndarray or a path to the directory containing the sequences and targets respectively")
        self.max_length = max_length
        self.min_length = min_length
        self.tokenizer = tokenizer
        self._check_data(task, remove_backbone_width, distance_threshold)

    def _check_data(self, task, remove_backbone_width, distance_threshold) -> None:
        print("Checking shape consistency...")
        input_data_dict = {'name': [], 'sequence': []}
        target_data_dict = {'name': [], 'target': []}
        
        for i, (input_seq, target) in enumerate(zip(self.input_lines, self.targets)):
            seq_len = len(input_seq)
            target_len = target.shape[0]
            target = torch.from_numpy(target)
            if remove_backbone_width > 0:
                target = remove_backbone_contacts(target, width=remove_backbone_width)

            if task == 'distance_map':
                target = bin_distances_2d(target)
                target = target.to(dtype=torch.float32)
            elif task == 'contact_map':
                if target.unique().shape[0] != 2:
                    target = target < distance_threshold
                    target = torch.nan_to_num(target, nan=-1)
                else:
                    print("Target is already a binary contact map. Skipping the conversion to contact map. Please make sure correct distance_threshold is used for binarization.")
            target = target.to(dtype=torch.long)

            values, _ = target.unique(return_counts=True)
            
            # Track issues for each sequence
            issues = []
            
            if seq_len < self.min_length:
                issues.append(f"length {seq_len} is less than {self.min_length}")
            if seq_len > self.max_length:
                issues.append(f"length {seq_len} is greater than {self.max_length}")
            if seq_len != target_len:
                issues.append(f"length {seq_len} doesn't match the target shape {target_len}")
            if task == 'distance_map':
                if len(values) <= 4:
                    issues.append(f"less than 4 classes in the target")
            
            # Handling issues
            if issues:
                name_info = self.individual_data_file_names[i] if self.individual_data_file_names else f"sequence_{i}"
                issues_str = "; ".join(issues)
                print(f"Discarding input files for {name_info} because: {issues_str}")
            else:
                name_info = self.individual_data_file_names[i] if self.individual_data_file_names else f"sequence_{i}"
                input_data_dict['name'].append(name_info)
                input_data_dict['sequence'].append(input_seq)
                target_data_dict['name'].append(name_info)
                target_data_dict['target'].append(target)
        
        self.input_data_df = pd.DataFrame(input_data_dict)
        self.target_data_df = pd.DataFrame(target_data_dict)
        
        self.input_lines = self.input_data_df['sequence'].tolist()
        self.targets = self.target_data_df['target'].tolist()


    def preprocess(self, sequence: str, idx)-> torch.Tensor:
        """
        Preprocesses the sequence for downstream training

        Args:
            sequence: The sequence to preprocess
            idx: The index of the sequence in the dataset
        Returns:
            input_ids: The input ids
            targets: The targets (usually contact/distance maps)
            seq_len: The length of the sequence

        """
        sequence = sequence.upper()
        sequence = sequence.replace('U', 'T') #! This is done because the upstream model is trained on RNA sequences with T instead of U
        sequence = sequence.replace('X', 'N') #! This is done to handle the unknown nucleotides
        sequence = sequence.replace('I', 'N') #! This is done to handle the unknown nucleotides
        input_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sequence))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_ids = torch.cat([torch.tensor([self.tokenizer.vocab['[CLS]']]), input_ids, torch.tensor([self.tokenizer.vocab['[SEP]']])], dim=0)
        targets = self.target_data_df['target'][idx]
        return input_ids, targets, len(sequence), self.target_data_df['name'][idx]


    def __len__(self)-> int:
        return len(self.input_lines)

    def __getitem__(self, idx)-> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        line = self.input_data_df['sequence'][idx]
        data = self.preprocess(line, idx)
        return data