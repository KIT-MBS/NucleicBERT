import os
import typing
import glob
import yaml
import torch
import numpy as np

class Config:
    def __init__(self, config) -> None:
        self.config = config
        self.model_config = self.config['model_config']
        self.train_data_config = self.config.get('train_data_config', None) # This can be None sometimes
        self.val_data_config = self.config.get('val_data_config', None) # This can be None sometimes
        self.test_data_config = self.config.get('test_data_config', None) # This can be None sometimes
        self.run_config = self.config['run_config']
        self.trainer_config = self.config['trainer_config']
        self.logging_config = self.config['logging_config']

    def __str__(self) -> str:
        return str(self.config)

def load_config(config_path: typing.Union[str, os.PathLike]) -> Config:
    with open(os.path.join(config_path)) as file:
        config_data = yaml.safe_load(file)

    return Config(config_data)

class CheckpointHandler:
    @staticmethod
    def find_latest_checkpoint(ckpt_dir: str, is_resume: bool = False) -> typing.Union[str, None]:
        ckpt_file = None
        if not is_resume:
            print("Starting fresh training...")
        elif is_resume:
            print(f'Looking for checkpoint file in the following directory: {ckpt_dir}')
            ckpt_files = glob.glob(os.path.join(ckpt_dir, '*.ckpt'))
            if ckpt_files:
                ckpt_file = max(ckpt_files, key=os.path.getctime)
                print(f"Checkpoint file found, resuming training from checkpoint: {ckpt_file}")
            else:
                ckpt_file = None
                print("Checkpoint file doesn't exist, this is likely an initial run. Starting fresh training...")

        return ckpt_file

def save_state_dict(dirpath, model) -> None:
    path = dirpath + '/state_dict.pth'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model state dict saved successfully at: {path}")


def remove_backbone_contacts(contacts: np.ndarray, width: int = 0):
    if isinstance(contacts, torch.Tensor):
        contacts = contacts.numpy()
        contacts = contacts.astype(np.float32)
    for i in range(1, width + 1):
        np.fill_diagonal(contacts[i:], np.nan)
        np.fill_diagonal(contacts[:, i:], np.nan)
        np.fill_diagonal(contacts[:-i],  np.nan)
        np.fill_diagonal(contacts[:, :-i], np.nan)

    # Make sure the diagonal is zero
    if width != 0:
        np.fill_diagonal(contacts, np.nan)

    return torch.from_numpy(contacts)

def bin_distances_2d(distances):
    """
    Categorize distances in a 2D matrix into 20 classes:
    - Class 0: distances < 2Å
    - Class 1-18: distances in [2Å, 20Å]
    - Class 19: distances > 20Å
    Args:
    distances (torch.Tensor): A 2D tensor (L x L) of pairwise distances.
    
    Returns:
    torch.Tensor: A 2D tensor of the same shape as `distances` with categorized values.
    """
    # Define the bin edges
    bin_edges = torch.linspace(2, 20, steps=19)
    
    # Add boundaries for distances < 2Å and distances > 20Å
    bin_edges = torch.cat((torch.tensor([-float('inf')]), bin_edges, torch.tensor([float('inf')])))
    
    # Use torch.bucketize to categorize the distances
    binned_distances = torch.bucketize(distances, bin_edges) - 1
    return binned_distances

def pad(
        input_tensor: torch.Tensor,
        max_length: int,
        pad_value: int = 0
    )-> torch.Tensor:
    """
    Pads the input tensor with pad_value to make it of length max_length
    
    Args:
        input_tensor: The input tensor to pad
        max_length: The length to pad the input tensor to
        pad_value: The value to pad the input tensor with
    Returns:
        padded_tensor: The padded tensor
        
    """
    pad_length = max_length - input_tensor.size(0)
    padding = torch.full(torch.Size([pad_length]), dtype=torch.long, fill_value=pad_value)
    padded_tensor = torch.cat((input_tensor, padding), dim=0)
    return padded_tensor

def pad_simple(
        input_ids,
        max_length,
        pad_value=0
    ):
    current_length = len(input_ids)
    pad_length = max_length - current_length
    padding = [pad_value] * pad_length
    padded_ids = input_ids + padding
    return padded_ids

def truncate(
        input: typing.Union[str, list],
        max_length
    )->typing.Union[str, list]:
    """
    Truncates the input to max_length

    Args:
        input: The input to truncate
        max_length: The maximum length of the input
    Returns:
        input: The truncated input
    """
    length = len(input)
    if length > max_length:
        input = input[:max_length]
    return input


def load_input_lines(input_dir: str)-> typing.List[str]:
    """
    Loads the input lines from the input_dir
    
    Args:
        input_dir: The path to the input directory
    Returns:
        input_lines: The list of input lines

    """
    if str(os.path.sep) not in input_dir:
        raise ValueError(f"Given input_dir {input_dir} is not a valid path. Please provide the full path to the input directory.")

    

    assert os.path.isdir(input_dir), f"Input directory {input_dir} does not exist"
    input_lines = []
    
    for file_name in sorted(os.listdir(input_dir)):
        file_path = os.path.join(input_dir, file_name)
        with open(file_path, 'r') as file:
            input_lines.extend(file.read().splitlines())
    return input_lines

def load_targets(target_dir: str)-> typing.List[np.ndarray]:
    """
    Loads the targets from the target_dir.
    The targets for the downstream task are binary contact matrices stored in .npy files.

    Args:
        target_dir: The path to the target directory
    Returns:
        targets: The list of targets

    """
    if str(os.path.sep) not in target_dir:
        raise ValueError(f"Given target_dir {target_dir} is not a valid path. Please provide the full path to the target directory.")
    assert os.path.isdir(target_dir), f"Target directory {target_dir} does not exist. In downstream mode, targets are required"
    target_file_list = sorted(os.listdir(target_dir))
    targets = [np.load(os.path.join(target_dir, file_name)) for file_name in target_file_list]
    return targets

