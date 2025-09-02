import os
import torch
import tqdm
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set_context('paper', font_scale=4.0)

from nucleicbert.models.bert import BERT, NB_CONFIG
from nucleicbert.downstream.closenessmodule import BERTWithClosenessResNet
from nucleicbert.inference.saliency.forward import SaliencyForward
from nucleicbert.inference.saliency.visualizers.sequence import set_style
# set_style()

import sys

class SaliencyMapGenerator:
    """
    Main generator class for creating saliency maps for different model types
    and performing analyses on them.
    """
    def __init__(
            self, 
            model_path: str,
            model_type: str,
            device: str = 'cpu',
            tokenizer = None, 
            model_config: Optional[Dict] = None,
        ):
        """
        Initialize the saliency map generator
        
        Args:
            model_path: Path to the saved model checkpoint
            model_type: Type of model ('bert', 'bert_with_secstr', 'bert_with_closeness')
            device: Device to run the model on
            tokenizer: Tokenizer to use for decoding
            model_config: Model configuration dictionary
        """
        self.device = device
        self.model_path = model_path
        self.model_type = model_type
        
        if model_config is None:
            model_config = NB_CONFIG
        self.model_config = model_config
        
        self.tokenizer = tokenizer
        
        # Load the model
        self.model = self._load_model(model_path, model_type)
        
        # Initialize saliency forward
        self.saliency_forward = SaliencyForward(
            self.model,
            model_type,
            device=device,
        )

    def _load_model(self, model_path: str, model_type: str) -> torch.nn.Module:
        """
        Load the pre-trained model from a checkpoint
        
        Args:
            model_path: Path to the model checkpoint
            model_type: Type of model
        
        Returns:
            Loaded model
        """
        if model_type == 'bert':
            model = BERT(**self.model_config)
            if model_path is not None:
                model.load_state_dict(torch.load(model_path, map_location=self.device))
        elif model_type == 'bert_with_secstr':
            # Load model config for secondary structure predictor
            model_config = {
                'vocab_size': self.model_config['vocab_size'],
                'hidden_size': self.model_config['hidden_size'],
                'num_hidden_layers': self.model_config['num_hidden_layers'],
                'num_attention_heads': self.model_config['num_attention_heads'],
                'dropout': self.model_config['dropout'],
                'max_length': self.model_config['max_length'],
                'position_embedding': self.model_config['position_embedding'],
            }
            
            run_config = {
                'lr': 5e-5,
                'weight_decay': 0.01,
                'max_epochs': 10,
                'internal_logs_freq': 100,
                'pretrained_path': None,  # We'll load the full model from checkpoint
                'frozen': False,  # This doesn't matter since we'll load the full model
                'use_scheduler': True,
            }
            
            # Initialize the model (need tokenizer for this)
            from transformers import PreTrainedTokenizerFast
            from nucleicbert.downstream.secstrmodule import SecondaryStructure2DPredictor
            model = SecondaryStructure2DPredictor(model_config, run_config, self.tokenizer)
            
            # Load the checkpoint
            if model_path is not None:
                checkpoint = torch.load(model_path, map_location=self.device)
                model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
        elif model_type == 'bert_with_closeness':
            bert = BERT(**NB_CONFIG)
            model = BERTWithClosenessResNet(
                bert,
                input_channels=NB_CONFIG['num_attention_heads']*NB_CONFIG['num_hidden_layers'],
                num_residual_blocks=1,
                task='contact_map'
            )
            if model_path is not None:
                model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model = model.to(self.device)
        model.eval()  # Set to evaluation mode
        return model

    def generate_salient_maps(
            self, 
            data_loader: torch.utils.data.DataLoader,
            save_dir: str,
            max_samples: int = 10,
            individual_visualizations: bool = True,
            run_aggregate_analysis: bool = False,
        ) -> Dict[str, torch.Tensor]:
        """
        Generate saliency maps for the input data and run analyses
        
        Args:
            data_loader: DataLoader for the input data
            save_dir: Directory to save the visualizations
            max_samples: Maximum number of samples to process
            individual_visualizations: Whether to create visualizations for individual samples
            run_aggregate_analysis: Whether to run aggregate analysis across all samples
        
        Returns:
            Dictionary of generated saliency maps
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Store saliency maps and other data
        saliency_maps = {}
        all_saliency_maps = []
        all_attn_weights = []
        all_contact_maps = []
        all_sequences = []
        all_structures = []
        
        for i, batch in enumerate(tqdm.tqdm(data_loader)):
            if i >= max_samples:
                break
                
            # Extract input_ids and targets from the batch
            if self.model_type == 'bert_with_secstr':
                input_ids, target_ids = batch[0], batch[1]  # input_ids and pair_matrix
            else:
                input_ids, target_ids = batch[0], batch[1]
            # Generate saliency map
            saliency_map, attn_weights = self.saliency_forward.forward_fn(input_ids, target_ids)
            saliency_maps[f'batch_{i}'] = saliency_map.detach().cpu()

            
            # Store for aggregate analysis
            for j in range(saliency_map.size(0)):
                seq_saliency = saliency_map[j].detach().cpu()
                seq_attn_weights = attn_weights[j].detach().cpu()
                all_saliency_maps.append(seq_saliency)
                all_attn_weights.append(seq_attn_weights)
                
                # For contact map model
                if self.model_type == 'bert_with_closeness' and len(batch) > 2:
                    contact_map = batch[1][j] if j < len(batch[1]) else None
                    if contact_map is not None:
                        all_contact_maps.append(contact_map.detach().cpu())
                
                # For secondary structure model
                if self.model_type == 'bert_with_secstr' and len(batch) > 2:
                    sequence = batch[1][j] if j < len(batch[1]) else None
                    structure = batch[-1][j] if j < len(batch[-1]) else None
                    if sequence is not None and structure is not None:
                        all_sequences.append(sequence)
                        all_structures.append(structure)
            
            # Individual sample visualizations
            if individual_visualizations:
                for j in range(saliency_map.size(0)):
                    if self.model_type == 'bert_with_secstr' and len(batch) > 2:
                        sequence = batch[-2][j] if j < len(batch[-2]) else None
                        structure_string = batch[-1][j] if j < len(batch[-1]) else None
                        structure_matrix = target_ids[j].cpu().numpy()  # Use the 2D pairing matrix from target_ids
                        
                        # Generate secondary structure visualizations
                        if sequence is not None and structure_string is not None:
                            self._visualize_secstr_saliency(
                                saliency_map[j], 
                                sequence, 
                                structure_matrix,  # Pass the 2D matrix instead of string
                                f"{save_dir}/saliency_structure_{i}_{j}.svg",
                                input_ids[j]
                            )
                            
                            # # Additional secondary structure analysis
                            # self._analyze_structure_vs_saliency(
                            #     saliency_map[j],
                            #     sequence,
                            #     structure,
                            #     f"{save_dir}/structure_analysis_{i}_{j}.png"
                            # )
                    
                    elif self.model_type == 'bert_with_closeness':
                        # For contact map model
                        if len(batch) > 2 and batch[1] is not None:  # Contact map is available
                            contact_map = batch[1][j] if j < len(batch[1]) else None
                            if contact_map is not None:
                                # # Basic contact map visualization
                                # self._visualize_contactmap_saliency(
                                #     saliency_map[j],
                                #     contact_map,
                                #     f"{save_dir}/saliency_contactmap_{i}_{j}.png",
                                #     input_ids[j]
                                # )
                                
                                # Correlation analysis
                                self._analyze_contactmap_correlation(
                                    saliency_map[j],
                                    contact_map,
                                    f"{save_dir}/contactmap_correlation_{i}_{j}.svg",
                                    input_ids[j]
                                )
                                
                                # # Detailed analysis
                                # self._analyze_contactmap_vs_saliency(
                                #     saliency_map[j],
                                #     contact_map,
                                #     f"{save_dir}/contactmap_analysis_{i}_{j}.png",
                                #     input_ids[j]
                                # )
                    
                    # Always create basic visualizations
                    self._visualize_basic_saliency(
                        saliency_map[j],
                        f"{save_dir}/saliency_basic_{i}_{j}.svg",
                        input_ids[j]
                    )
        
        # Run aggregate analysis if requested and we have data
        if run_aggregate_analysis:
            # Create directory for aggregate analyses
            aggregate_dir = os.path.join(save_dir, 'aggregate')
            os.makedirs(aggregate_dir, exist_ok=True)
            
            if self.model_type == 'bert_with_closeness' and all_contact_maps:
                print(f"Running aggregate analysis on {len(all_saliency_maps)} samples...")
                
                # Import aggregation functions
                from nucleicbert.inference.saliency.visualizers.aggregate import run_all_aggregate_analyses
                
                # Run all aggregate analyses
                results = run_all_aggregate_analyses(
                    all_saliency_maps,
                    all_contact_maps,
                    aggregate_dir,
                    prefix='contact_maps'
                )
                
                print(f"Aggregate analysis complete. Results saved to {aggregate_dir}")
            
            elif self.model_type == 'bert_with_secstr' and all_sequences and all_structures:
                print(f"Running secondary structure aggregate analysis on {len(all_saliency_maps)} samples...")
                
                # Import new secondary structure analysis functions
                from nucleicbert.inference.saliency.visualizers.secstr_aggregate import (
                    run_all_secstr_analyses,
                    compare_saliency_across_rna_properties
                )
                
                # Run comprehensive secondary structure analysis
                secstr_results = run_all_secstr_analyses(
                    all_saliency_maps,
                    all_sequences,
                    all_structures,
                    aggregate_dir,
                    prefix='secstr'
                )
                
                # Run RNA property comparison
                rna_prop_results = compare_saliency_across_rna_properties(
                    all_saliency_maps,
                    all_sequences,
                    all_structures,
                    aggregate_dir,
                    prefix='rna_properties'
                )
                
                print(f"Secondary structure aggregate analysis complete. Results saved to {aggregate_dir}")
            
            else:
                # Basic saliency aggregation for any model type
                from nucleicbert.inference.saliency.visualizers.aggregate import aggregate_saliency_maps
                
                print(f"Running basic saliency aggregation on {len(all_saliency_maps)} samples...")
                aggregate_saliency_maps(
                    all_saliency_maps,
                    aggregate_dir,
                    prefix='saliency'
                )
                
                print(f"Basic aggregation complete. Results saved to {aggregate_dir}")
            
            # # Create master index.html for aggregate analysis
            # self._create_master_index(save_dir, self.model_type)
        
        # Save all saliency maps for external analysis
        torch.save(all_saliency_maps, os.path.join(save_dir, 'all_saliency_maps.pt'))
        torch.save(all_attn_weights, os.path.join(save_dir, 'all_attn_weights.pt'))
        
        if all_contact_maps:
            torch.save(all_contact_maps, os.path.join(save_dir, 'all_contact_maps.pt'))
        
        if all_sequences and all_structures:
            np.save(os.path.join(save_dir, 'all_sequences_structures.npy'), 
                   {'sequences': all_sequences, 'structures': all_structures})
        
        return saliency_maps
    
    def _visualize_basic_saliency(self, saliency_map, save_path, input_ids=None):
        """
        Create basic saliency visualization
        """
        from nucleicbert.inference.saliency.visualizers.sequence import plot_improved_saliency
        return plot_improved_saliency(
            saliency_map, 
            save_path,
            input_ids=input_ids,
            tokenizer=self.tokenizer
        )
    
    def _visualize_secstr_saliency(self, saliency_map, sequence, structure, save_path, input_ids=None):
        """
        Create secondary structure saliency visualization
        """
        from nucleicbert.inference.saliency.visualizers.secstr import plot_saliency_structure_correlation
        # set_style()  # Ensure consistent style for secondary structure plots
        return plot_saliency_structure_correlation(
            saliency_map, 
            sequence, 
            structure, 
            save_path,
        )
    
    # def _analyze_structure_vs_saliency(self, saliency_map, sequence, structure, save_path):
    #     """
    #     Perform detailed statistical analysis of the relationship between
    #     RNA secondary structure and saliency values
    #     """
    #     from nucleicbert.inference.saliency.visualizers.secstr import analyze_structure_vs_saliency_fixed
    #     return analyze_structure_vs_saliency_fixed(
    #         saliency_map,
    #         sequence,
    #         structure,
    #         save_path
    #     )
    
    # def _visualize_contactmap_saliency(self, saliency_map, contact_map, save_path, input_ids=None):
    #     """
    #     Create contact map saliency visualization
    #     """
    #     from nucleicbert.inference.saliency.visualizers.contactmap import plot_saliency_with_contactmap
    #     return plot_saliency_with_contactmap(
    #         saliency_map, 
    #         contact_map, 
    #         save_path,
    #         input_ids=input_ids,
    #         tokenizer=self.tokenizer
    #     )
    
    def _analyze_contactmap_correlation(self, saliency_map, contact_map, save_path, input_ids=None):
        """
        Analyze correlation between saliency and contact map
        """
        from nucleicbert.inference.saliency.visualizers.contactmap import plot_saliency_contactmap_correlation
        return plot_saliency_contactmap_correlation(
            saliency_map,
            contact_map,
            save_path,
            input_ids=input_ids,
            tokenizer=self.tokenizer
        )
    
    # def _analyze_contactmap_vs_saliency(self, saliency_map, contact_map, save_path, input_ids=None):
    #     """
    #     Detailed analysis of contact map properties vs saliency
    #     """
    #     from nucleicbert.inference.saliency.visualizers.contactmap import analyze_contactmap_vs_saliency
    #     return analyze_contactmap_vs_saliency(
    #         saliency_map,
    #         contact_map,
    #         save_path,
    #         input_ids=input_ids,
    #         tokenizer=self.tokenizer
    #     )