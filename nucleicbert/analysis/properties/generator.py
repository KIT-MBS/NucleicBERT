import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
import re
import csv


class PropertyGenerator:
    """
    Generates property maps for attention analysis from model outputs.
    Works with the SaliencyForward class to extract both saliency maps and attention weights,
    and creates property maps that can be used with attention analysis code.
    """
    def __init__(
            self,
            model,
            model_type: str,
            device: str = 'cuda',
            tokenizer = None,
            motifs_csv_path: Optional[str] = None,
        ):
        """
        Initialize the property generator
        
        Args:
            model: The model to generate properties for
            model_type: Type of model ('bert', 'bert_with_secstr', 'bert_with_closeness', 
                                      'bert_with_contactmap', 'bert_with_shuffle')
            device: Device to run the model on
            tokenizer: Tokenizer to use for decoding
            motifs_csv_path: Path to CSV file containing RNA motifs
        """
        self.model = model
        self.model_type = model_type
        self.device = device
        self.tokenizer = tokenizer
        

        from nucleicbert.inference.saliency.forward import SaliencyForward
        self.saliency_forward = SaliencyForward(
            self.model,
            model_type,
            device=device,
        )
        
        # Load RNA motifs from CSV file if provided
        self.rna_motifs = {}
        if motifs_csv_path is not None:
            self._load_motifs_from_csv(motifs_csv_path)
        else:
            # Use default motifs if no CSV provided
            self.rna_motifs = {
                'hairpin_loop': r'G[ACGU]{3,8}C',      # Simple hairpin loop pattern
                'bulge': r'[ACGU]{2,4}G[ACGU]{5,10}C', # Bulge pattern
                'tetraloop': r'GNRA',                  # GNRA tetraloop
                'k_turn': r'GA.{2,3}G',                # K-turn motif simplified
                'g_quadruplex': r'G{3,5}.{1,7}G{3,5}', # G-quadruplex simplified pattern
            }
    
    def _load_motifs_from_csv(self, csv_path: str) -> None:
        """
        Load RNA motifs from a CSV file
        
        Args:
            csv_path: Path to the CSV file containing motifs
        """
        try:

            motifs_df = pd.read_csv(csv_path)
            

            if 'Motif' in motifs_df.columns:
                unique_motifs = motifs_df['Motif'].unique()
                

                for motif in unique_motifs:
                    if isinstance(motif, str) and len(motif) > 0:
                        self.rna_motifs[motif] = re.escape(motif)  
            
            print(f"Loaded {len(self.rna_motifs)} unique RNA motifs from {csv_path}")
            
        except Exception as e:
            print(f"Error loading motifs from CSV: {e}")
            # Fall back to default motifs
            self.rna_motifs = {
                'hairpin_loop': r'G[ACGU]{3,8}C',
                'bulge': r'[ACGU]{2,4}G[ACGU]{5,10}C',
                'tetraloop': r'GNRA',
                'k_turn': r'GA.{2,3}G',
                'g_quadruplex': r'G{3,5}.{1,7}G{3,5}',
            }
    
    def generate_properties(
            self,
            data_loader: torch.utils.data.DataLoader,
            max_samples: int = 10,
        ) -> Tuple[List[torch.Tensor], Dict[str, List[np.ndarray]], List[str]]:
        """
        Generate attention maps and property maps for analysis
        
        Args:
            data_loader: DataLoader for the input data
            max_samples: Maximum number of samples to process
        
        Returns:
            Tuple containing:
                - List of attention maps [num_samples, num_layers, num_heads, seq_len, seq_len]
                - Dictionary of property maps {property_name: [num_samples, seq_len] or [num_samples, seq_len, seq_len]}
                - List of property names
        """
        attention_maps = []
        properties = {}
        
        for i, batch in enumerate(data_loader):
            if i >= max_samples:
                break
                
            if self.model_type == 'bert_with_secstr':
                input_ids, target_ids = batch[0], batch[1]  # input_ids and target_matrix
            else:
                input_ids, target_ids = batch[0], batch[1]
            
            outputs = self.saliency_forward.forward_fn(input_ids, target_ids)
            saliency_map = outputs[0]
            attn_weights = outputs[1].squeeze().detach().cpu()
            

            attention_maps.append(attn_weights)
            

            if i == 0:
                if self.model_type == 'bert_with_secstr':
                    self._init_secondary_structure_properties(properties)
                elif self.model_type == 'bert_with_closeness' or self.model_type == 'bert_with_contactmap':
                    self._init_contact_map_properties(properties)                

                if self.model_type in ['bert_with_secstr', 'bert_with_closeness', 
                                       'bert_with_contactmap', 'bert_with_shuffle']:
                    if 'motif_recognition' not in properties:
                        properties['motif_recognition'] = {}
            

            if self.model_type == 'bert_with_secstr':
                self._extract_secondary_structure_properties(batch, properties, i)
            elif self.model_type == 'bert_with_closeness' or self.model_type == 'bert_with_contactmap':
                self._extract_contact_map_properties(batch, properties, i)            

            if self.model_type in ['bert_with_secstr', 'bert_with_closeness', 
                                   'bert_with_contactmap', 'bert_with_shuffle']:
                self._extract_motif_recognition_property(batch, properties, i)
        

        property_lists = {}
        for prop_name, prop_values in properties.items():
            property_lists[prop_name] = [prop_values[j] for j in range(len(prop_values))]
        
        property_names = list(property_lists.keys())
        
        return attention_maps, property_lists, property_names
    
    def _init_secondary_structure_properties(self, properties: Dict[str, Dict[int, np.ndarray]]):
        """Initialize properties specific to secondary structure prediction using binary matrix"""
        properties['paired_positions'] = {}      # Binary: 1 for paired, 0 for unpaired
        properties['contact_density'] = {}       # Number of contacts per position from binary matrix
    
    def _init_contact_map_properties(self, properties: Dict[str, Dict[int, np.ndarray]]):
        """Initialize properties specific to contact map prediction"""
        properties['contact_degree'] = {}     # Number of contacts per position
        properties['long_range_ratio'] = {}   # Ratio of long-range contacts (>24 positions away)

    
    def _extract_secondary_structure_properties(
            self, 
            batch: Tuple, 
            properties: Dict[str, Dict[int, np.ndarray]], 
            idx: int
        ):
        """Extract secondary structure properties from binary matrix"""

        if len(batch) > 1 and batch[1] is not None:
            structure_matrices = batch[1].cpu().numpy()
            
            for b in range(structure_matrices.shape[0]):
                structure_matrix = structure_matrices[b]
                seq_len = structure_matrix.shape[0]
                

                paired_positions = np.zeros(seq_len)
                for i in range(seq_len):
                    if np.any(structure_matrix[i, :]):
                        paired_positions[i] = 1
                
                # Contact density: number of contacts per position
                contact_density = np.sum(structure_matrix, axis=1).astype(np.float32)
                
                # Pad properties to match attention map dimensions (add CLS and SEP positions)
                # Get input_ids to determine the full sequence length including special tokens
                input_ids = batch[0]
                full_seq_len = input_ids.shape[1]  # Includes CLS and SEP
                
                # Pad paired_positions: CLS=0, actual_sequence, SEP=0
                padded_paired_positions = np.zeros(full_seq_len)
                padded_paired_positions[1:seq_len+1] = paired_positions  # Skip CLS (index 0), add SEP padding
                
                # Pad contact_density: CLS=0, actual_sequence, SEP=0  
                padded_contact_density = np.zeros(full_seq_len)
                padded_contact_density[1:seq_len+1] = contact_density  # Skip CLS (index 0), add SEP padding
                
                properties['paired_positions'][idx * structure_matrices.shape[0] + b] = padded_paired_positions
                properties['contact_density'][idx * structure_matrices.shape[0] + b] = padded_contact_density
    
    def _extract_contact_map_properties(
            self, 
            batch: Tuple, 
            properties: Dict[str, Dict[int, np.ndarray]], 
            idx: int
        ):
        """Extract contact map properties for analysis"""

        if len(batch) > 2 and batch[1] is not None:
            contact_maps = batch[1].cpu().numpy()
            
            for b in range(contact_maps.shape[0]):
                contact_map = contact_maps[b]
                seq_len = contact_map.shape[0]
                
                # Assuming all contact maps are binary (1 for contact, 0 for no contact)
                binary_contact = contact_map.astype(np.int32)
                
                # 1. Contact degree: number of contacts per position
                contact_degree = np.sum(binary_contact, axis=1).astype(np.float32)
                properties['contact_degree'][idx * contact_maps.shape[0] + b] = contact_degree
                
                # 2. Long-range contact ratio
                long_range_threshold = 24  # Positions
                long_range_contacts = np.zeros(seq_len)
                short_range_contacts = np.zeros(seq_len)
                
                for pos in range(seq_len):
                    for j in range(seq_len):
                        if binary_contact[pos, j] == 1:
                            if abs(pos - j) > long_range_threshold:
                                long_range_contacts[pos] += 1
                            else:
                                short_range_contacts[pos] += 1
                
                # Calculate ratio, avoiding division by zero
                total_contacts = long_range_contacts + short_range_contacts
                long_range_ratio = np.zeros(seq_len)
                mask = total_contacts > 0
                long_range_ratio[mask] = long_range_contacts[mask] / total_contacts[mask]
                properties['long_range_ratio'][idx * contact_maps.shape[0] + b] = long_range_ratio

    
    def _extract_motif_recognition_property(
            self, 
            batch: Tuple, 
            properties: Dict[str, Dict[int, np.ndarray]], 
            idx: int
        ):
        """Extract motif recognition property using sequence from batch[1] for secstr model"""
        
        if self.model_type == 'bert_with_secstr':

            if len(batch) > 2:
                sequences = batch[1]
            else:
                sequences = None
                
            if sequences is None:
                print("WARNING: No sequences found in batch for motif recognition")
                # Fallback: create dummy sequences
                input_ids = batch[0].cpu().numpy()
                sequences = ["A" * input_ids.shape[1]] * input_ids.shape[0]
        else:

            input_ids = batch[0].cpu().numpy()
            sequences = []
            
            for b in range(input_ids.shape[0]):
                if self.tokenizer:

                    seq_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[b])
                    

                    filtered_tokens = [t for t in seq_tokens if t not in ['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]']]
                    

                    if '##' in ''.join(seq_tokens):

                        clean_tokens = [t.replace('##', '') for t in filtered_tokens]
                        sequence = ''.join(clean_tokens)
                    elif filtered_tokens and any(t.startswith('Ġ') for t in filtered_tokens):

                        clean_tokens = [t.replace('Ġ', '') for t in filtered_tokens]
                        sequence = ''.join(clean_tokens)
                    else:

                        sequence = ''.join(filtered_tokens)
                        

                        if not all(c in 'ACGTU acgtu' for c in sequence if c.isalpha()):
                            sequence = ' '.join(filtered_tokens)
                else:

                    sequence = "A" * input_ids.shape[1]  # Dummy sequence of appropriate length
                
                sequences.append(sequence)
        

        for b, sequence in enumerate(sequences):

            sequence = str(sequence).upper()  # Convert to uppercase for consistency
            

            seq_len = len(sequence)
            

            motif_recognition = np.zeros(seq_len)
            
            if isinstance(sequence, str) and seq_len > 0:

                for motif_name, pattern in self.rna_motifs.items():
                    try:

                        if motif_name in sequence:
                            start_idx = sequence.find(motif_name)
                            while start_idx != -1:
                                end_idx = start_idx + len(motif_name)
                                if end_idx <= seq_len:  # Ensure within bounds
                                    motif_recognition[start_idx:end_idx] = 1
                                # Find next occurrence
                                start_idx = sequence.find(motif_name, end_idx)
                        
                        # Then try regex matching
                        for match in re.finditer(pattern, sequence):
                            start, end = match.span()
                            if start < seq_len and end <= seq_len:  # Ensure within bounds
                                # Mark the positions where motifs occur
                                motif_recognition[start:end] = 1
                    except re.error:
                        # Log invalid regex patterns
                        continue
            
            # Pad motif recognition to match attention map dimensions (add CLS and SEP positions)
            # Get input_ids to determine the full sequence length including special tokens
            input_ids = batch[0]
            full_seq_len = input_ids.shape[1]  # Includes CLS and SEP
            
            # Pad motif_recognition: CLS=0, actual_sequence, SEP=0
            padded_motif_recognition = np.zeros(full_seq_len)
            if seq_len > 0:
                # Ensure we don't exceed bounds
                copy_len = min(seq_len, full_seq_len - 2)  # Leave room for CLS and SEP
                padded_motif_recognition[1:1+copy_len] = motif_recognition[:copy_len]  # Skip CLS (index 0)
            
            properties['motif_recognition'][idx * len(sequences) + b] = padded_motif_recognition
                
    def save_properties(
            self, 
            attention_maps: List[torch.Tensor],
            property_maps: Dict[str, List[np.ndarray]],
            property_names: List[str],
            output_dir: str
        ):
        """
        Save attention maps and property maps to files
        
        Args:
            attention_maps: List of attention maps
            property_maps: Dictionary of property maps
            property_names: List of property names
            output_dir: Directory to save files
        """
        import pandas as pd
        os.makedirs(output_dir, exist_ok=True)
        

        torch.save(attention_maps, os.path.join(output_dir, 'attention_maps.pt'))
        

        props_dir = os.path.join(output_dir, 'properties')
        os.makedirs(props_dir, exist_ok=True)
        

        for prop_name in property_names:
            prop_dir = os.path.join(props_dir, prop_name)
            os.makedirs(prop_dir, exist_ok=True)
            

            for sample_idx, prop_array in enumerate(property_maps[prop_name]):

                if prop_array.ndim == 1:  # 1D property
                    df = pd.DataFrame(prop_array, columns=['value'])
                else:  # 2D property
                    df = pd.DataFrame(prop_array)

                csv_path = os.path.join(prop_dir, f'sample_{sample_idx}.csv')
                df.to_csv(csv_path, index=False)
        

        with open(os.path.join(output_dir, 'property_names.txt'), 'w') as f:
            for name in property_names:
                f.write(f"{name}\n")