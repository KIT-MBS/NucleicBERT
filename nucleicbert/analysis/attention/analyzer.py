import os
import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import normalize
import seaborn as sns

from nucleicbert.analysis.properties.generator import PropertyGenerator

sns.set_context("paper", font_scale=2.0)

import juelich_colors

def set_style():
    font = {
            'family' : 'sans-serif',
            # 'weight' : 'bold',
            'size'   : 25
    }
    axes = {
            'titlesize' : 25,
            'labelsize' : 25,
            # 'labelweight' : 'bold',
            # 'titleweight' : 'bold'
    }
    xtick = {
            'labelsize' : 25,
    }
    ytick = {
            'labelsize' : 25,
    }
    legend = {
            'fontsize' : 15,
            'title_fontsize' : 15,
            'markerscale': 2,
    }
    lines = {
            'markersize': 10,
    }
    
    matplotlib.rc('font', **font)
    matplotlib.rc('axes', **axes)
    matplotlib.rc('xtick', **xtick)
    matplotlib.rc('ytick', **ytick)
    matplotlib.rc('legend', **legend)
    matplotlib.rc('lines', **lines)

def create_juelich_colormap():
    """Create custom colormap using Jülich colors"""
    colors = [
        juelich_colors.custom_colors['light_grey'],
        juelich_colors.custom_colors['julich_blue_2'], 
        juelich_colors.custom_colors['julich_blue_1'],
        juelich_colors.custom_colors['raspberry']
    ]
    return LinearSegmentedColormap.from_list("juelich", colors, N=256)

class AttentionMapAnalyzer:
    """
    Main class for analyzing attention maps to understand how different layers and heads
    contribute to the prediction of various properties in RNA/DNA sequences.
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
        Initialize the attention map analyzer
        
        Args:
            model: The model to analyze
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
        self.motifs_csv_path = motifs_csv_path
        
        self.property_generator = PropertyGenerator(
            model,
            model_type,
            device=device,
            tokenizer=tokenizer,
            motifs_csv_path=motifs_csv_path,
        )
    
    def analyze_attention_for_data(
            self,
            data_loader: torch.utils.data.DataLoader,
            output_dir: str,
            max_samples: int = 10,
            input_dir: Optional[str] = None,
            custom_properties: Optional[Dict[str, List[np.ndarray]]] = None,
            custom_property_names: Optional[List[str]] = None,
        ):
        """
        Analyze attention maps for the data and save results
        
        Args:
            data_loader: DataLoader for the input data
            output_dir: Directory to save the results
            max_samples: Maximum number of samples to process
            input_dir: Optional directory to read pre-computed CSV files
            custom_properties: Optional custom property maps to analyze
            custom_property_names: Optional custom property names for the custom maps
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if custom_properties is None or custom_property_names is None:
            attention_maps, properties, property_names = self.property_generator.generate_properties(
                data_loader,
                max_samples=max_samples,
            )
        else:
            attention_maps, properties, property_names = self._prepare_custom_properties(
                data_loader,
                custom_properties,
                custom_property_names,
                max_samples=max_samples,
            )
        
        prop_dir = os.path.join(output_dir, 'properties')
        os.makedirs(prop_dir, exist_ok=True)
        self.property_generator.save_properties(
            attention_maps,
            properties,
            property_names,
            prop_dir,
        )

        threshold_dir = os.path.join(output_dir, 'threshold_analysis')
        os.makedirs(threshold_dir, exist_ok=True)
        self._analyze_threshold_based(
            attention_maps,
            properties,
            property_names,
            threshold_dir,
            input_dir,
        )
        

        topL_dir = os.path.join(output_dir, 'topL_analysis')
        os.makedirs(topL_dir, exist_ok=True)
        self._analyze_topL_based(
            attention_maps,
            properties,
            property_names,
            topL_dir,
            input_dir,
        )
        

        layer_dir = os.path.join(output_dir, 'layer_analysis')
        os.makedirs(layer_dir, exist_ok=True)
        self._analyze_layer_aggregated(
            attention_maps,
            properties,
            property_names,
            layer_dir,
        )
        

        head_dir = os.path.join(output_dir, 'head_analysis')
        os.makedirs(head_dir, exist_ok=True)
        self._analyze_head_aggregated(
            attention_maps,
            properties,
            property_names,
            head_dir,
        )
        

        if 'motif_recognition' in property_names:
            motif_dir = os.path.join(output_dir, 'motif_analysis')
            os.makedirs(motif_dir, exist_ok=True)
            self._analyze_motif_attention(
                attention_maps,
                properties,
                motif_dir,
            )
    
    def _prepare_custom_properties(
            self,
            data_loader: torch.utils.data.DataLoader,
            custom_properties: Dict[str, List[np.ndarray]],
            custom_property_names: List[str],
            max_samples: int = 10,
        ) -> Tuple[List[torch.Tensor], Dict[str, List[np.ndarray]], List[str]]:
        """
        Prepare custom properties and extract attention maps
        
        Args:
            data_loader: DataLoader for the input data
            custom_properties: Custom property maps to analyze
            custom_property_names: Custom property names
            max_samples: Maximum number of samples to process
        
        Returns:
            Tuple containing attention maps, properties, and property names
        """
        attention_maps = []
        
        for i, batch in enumerate(data_loader):
            if i >= max_samples:
                break
                
  
            input_ids, target_ids = batch[0], batch[1]
            
            outputs = self.property_generator.saliency_forward.forward_fn(input_ids, target_ids)
            attn_weights = outputs[1].detach().cpu()
            
            attention_maps.append(attn_weights)
        
        return attention_maps, custom_properties, custom_property_names
    
    def _analyze_threshold_based(
            self,
            attention_maps: List[torch.Tensor],
            properties: Dict[str, List[np.ndarray]],
            property_names: List[str],
            output_dir: str,
            input_dir: Optional[str] = None,
        ):
        """
        Perform threshold-based analysis of attention maps
        
        Args:
            attention_maps: List of attention maps
            properties: Dictionary of property maps
            property_names: List of property names
            output_dir: Directory to save the results
            input_dir: Optional directory to read pre-computed CSV files
        """
        from nucleicbert.analysis.attention_analysis import plot_attention_property_heatmaps_threshold
        
        property_lists = []
        for prop_name in property_names:
            property_lists.append(properties[prop_name])
        

        plot_attention_property_heatmaps_threshold(
            attention_maps,
            property_lists,
            property_names,
            output_dir,
            input_dir,
        )
    
    def _analyze_topL_based(
            self,
            attention_maps: List[torch.Tensor],
            properties: Dict[str, List[np.ndarray]],
            property_names: List[str],
            output_dir: str,
            input_dir: Optional[str] = None,
        ):
        """
        Perform top-L based analysis of attention maps
        
        Args:
            attention_maps: List of attention maps
            properties: Dictionary of property maps
            property_names: List of property names
            output_dir: Directory to save the results
            input_dir: Optional directory to read pre-computed CSV files
        """
        from nucleicbert.analysis.attention_analysis import plot_attention_property_heatmaps_topL
        
        property_lists = []
        for prop_name in property_names:
            property_lists.append(properties[prop_name])
        
        plot_attention_property_heatmaps_topL(
            attention_maps,
            property_lists,
            property_names,
            output_dir,
            input_dir,
        )
    
    def _analyze_layer_aggregated(
            self,
            attention_maps: List[torch.Tensor],
            properties: Dict[str, List[np.ndarray]],
            property_names: List[str],
            output_dir: str,
        ):
        """
        Perform layer-wise aggregated analysis of attention maps
        
        Args:
            attention_maps: List of attention maps
            properties: Dictionary of property maps
            property_names: List of property names
            output_dir: Directory to save the results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # We assume that all attention maps share the same shape
        num_layers, num_heads = attention_maps[0].shape[:2]
        

        for prop_idx, prop_name in enumerate(property_names):
            prop_maps = properties[prop_name]
            

            layer_metrics = np.zeros(num_layers)
            layer_counts = np.zeros(num_layers)
            

            for sample_idx in range(len(attention_maps)):
                att_map = attention_maps[sample_idx]
                att_map = torch.nn.functional.normalize(att_map, dim=-1)  # Normalize attention if needed
                att_map = att_map.numpy() if hasattr(att_map, "numpy") else np.array(att_map)
                prop_map = prop_maps[sample_idx]
                
                # For 1D properties, expand to 2D
                if prop_map.ndim == 1:
                    seq_length = att_map.shape[3]  # attention shape [batch, layers, heads, seq_len, seq_len]
                    prop_map_resized = np.resize(prop_map, seq_length)
                    prop_mat = np.tile(prop_map_resized, (seq_length, 1))
                else:
                    prop_mat = prop_map

                for l in range(num_layers):

                    layer_att = np.mean(att_map[l], axis=0)  # [seq_len, seq_len]
                    

                    if np.unique(prop_mat).shape[0] <= 2:  # Binary property
                        high_att_mask = layer_att > np.percentile(layer_att, 90)  # Top 10% attention
                        metric = np.mean(prop_mat[high_att_mask])
                    else:  # Continuous property
                        flat_att = layer_att.flatten()
                        flat_prop = prop_mat.flatten()
                        
                        valid_mask = ~np.isnan(flat_prop) & (flat_prop != 0)
                        if np.sum(valid_mask) > 0:
                            correlation = np.corrcoef(flat_att[valid_mask], flat_prop[valid_mask])[0, 1]
                            metric = correlation if not np.isnan(correlation) else 0
                        else:
                            metric = 0
                    
                    layer_metrics[l] += metric
                    layer_counts[l] += 1
            

            avg_metrics = np.divide(
                layer_metrics, 
                layer_counts, 
                out=np.zeros_like(layer_metrics), 
                where=(layer_counts != 0)
            )
            
            df = pd.DataFrame({
                'layer': np.arange(num_layers),
                'metric': avg_metrics,
                'count': layer_counts
            })
            df.to_csv(os.path.join(output_dir, f'layer_aggregated_{prop_name}.csv'), index=False)
            

            plt.figure(figsize=(10, 5))
            bars = plt.bar(np.arange(num_layers), avg_metrics, color=juelich_colors.custom_colors['julich_blue_1'], alpha=0.8, edgecolor=juelich_colors.custom_colors['julich_blue_2'], linewidth=2)
            plt.xlabel('Layer')
            plt.ylabel('Metric Value')
            plt.title(f'Layer-wise Correlation with {prop_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'layer_aggregated_{prop_name}.svg'), dpi=300, transparent=True, format='svg')
            plt.close()
    
    def _analyze_head_aggregated(
            self,
            attention_maps: List[torch.Tensor],
            properties: Dict[str, List[np.ndarray]],
            property_names: List[str],
            output_dir: str,
        ):
        """
        Perform head-wise aggregated analysis of attention maps
        
        Args:
            attention_maps: List of attention maps
            properties: Dictionary of property maps
            property_names: List of property names
            output_dir: Directory to save the results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        num_layers, num_heads = attention_maps[0].shape[:2]
        
        for prop_idx, prop_name in enumerate(property_names):
            prop_maps = properties[prop_name]
            
            # Create per-head metrics
            head_metrics = np.zeros(num_heads)
            head_counts = np.zeros(num_heads)
            
            # For each sample
            for sample_idx in range(len(attention_maps)):
                att_map = attention_maps[sample_idx]
                att_map = torch.nn.functional.normalize(att_map, dim=-1)  # Normalize attention
                att_map = att_map.numpy() if hasattr(att_map, "numpy") else np.array(att_map)
                prop_map = prop_maps[sample_idx]
                

                if prop_map.ndim == 1:
                    seq_length = att_map.shape[3]  # attention shape [batch, layers, heads, seq_len, seq_len]
                    prop_map_resized = np.resize(prop_map, seq_length)
                    # Create a matrix by repeating this vector for each row
                    prop_mat = np.tile(prop_map_resized, (seq_length, 1))
                else:
                    prop_mat = prop_map
                
                for h in range(num_heads):
                    head_att = np.mean(att_map[:, h, :, :], axis=0)  # [seq_len, seq_len]
                    
                    if np.unique(prop_mat).shape[0] <= 2:  # Binary property
                        high_att_mask = head_att > np.percentile(head_att, 90)  # Top 10% attention
                        metric = np.mean(prop_mat[high_att_mask])
                    else:  # Continuous property
                        flat_att = head_att.flatten()
                        flat_prop = prop_mat.flatten()
                        

                        valid_mask = ~np.isnan(flat_prop) & (flat_prop != 0)
                        if np.sum(valid_mask) > 0:
                            correlation = np.corrcoef(flat_att[valid_mask], flat_prop[valid_mask])[0, 1]
                            metric = correlation if not np.isnan(correlation) else 0
                        else:
                            metric = 0
                    
                    head_metrics[h] += metric
                    head_counts[h] += 1
            
            avg_metrics = np.divide(
                head_metrics, 
                head_counts, 
                out=np.zeros_like(head_metrics), 
                where=(head_counts != 0)
            )
            

            df = pd.DataFrame({
                'head': np.arange(num_heads),
                'metric': avg_metrics,
                'count': head_counts
            })
            df.to_csv(os.path.join(output_dir, f'head_aggregated_{prop_name}.csv'), index=False)
            

            plt.figure(figsize=(10, 5))
            bars = plt.bar(np.arange(num_heads), avg_metrics, color=juelich_colors.custom_colors['grass_green'], alpha=0.8, edgecolor=juelich_colors.custom_colors['julich_blue_1'], linewidth=2)
            plt.xlabel('Head')
            plt.ylabel('Metric Value')
            plt.title(f'Head-wise Correlation with {prop_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'head_aggregated_{prop_name}.svg'), dpi=300, transparent=True, format='svg')
            plt.close()
    
    def _analyze_motif_attention(
            self,
            attention_maps: List[torch.Tensor],
            properties: Dict[str, List[np.ndarray]],
            output_dir: str,
        ):
        """
        Analyze attention patterns for motif recognition
        
        Args:
            attention_maps: List of attention maps
            properties: Dictionary of property maps
            output_dir: Directory to save the results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        prop_maps = properties['motif_recognition']
        
        num_layers, num_heads = attention_maps[0].shape[:2]
        
        motif_attention_heatmap = np.zeros((num_layers, num_heads))
        sample_count = 0
        
        for sample_idx in range(len(attention_maps)):
            att_map = attention_maps[sample_idx]
            att_map = torch.nn.functional.normalize(att_map, dim=-1)  # Normalize attention
            att_map = att_map.numpy() if hasattr(att_map, "numpy") else np.array(att_map)
            motif_map = prop_maps[sample_idx]
            
            # Skip if no motifs found
            if np.sum(motif_map) == 0:
                continue
                
            sample_count += 1
            
            motif_positions = np.where(motif_map > 0)[0]
            

            for l in range(num_layers):
                for h in range(num_heads):
                    head_att = att_map[l, h]  # [seq_len, seq_len]
                    
                    motif_attention = 0
                    
                    if len(motif_positions) > 0:
                        att_to_motifs = head_att[:, motif_positions].mean()
                        att_from_motifs = head_att[motif_positions].mean()
                        motif_attention = max(att_to_motifs, att_from_motifs)
                    
                    motif_attention_heatmap[l, h] += motif_attention
        
        if sample_count > 0:
            motif_attention_heatmap /= sample_count
        
        plt.figure(figsize=(7, 5))
        plt.imshow(motif_attention_heatmap, cmap=create_juelich_colormap(), aspect='auto')
        plt.colorbar(label='Average Attention to Motifs')
        plt.xlabel('Head')
        plt.ylabel('Layer')
        plt.title('Attention to RNA Motifs per Layer and Head')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'motif_attention_heatmap.svg'), dpi=300, transparent=True, format='svg')
        plt.close()
        
        motif_df = pd.DataFrame(motif_attention_heatmap)
        motif_df.to_csv(os.path.join(output_dir, 'motif_attention_heatmap.csv'), index=False)
        
        flat_idx = np.argsort(motif_attention_heatmap.flatten())[::-1][:10]  # Top 10 indices
        layers, heads = np.unravel_index(flat_idx, motif_attention_heatmap.shape)
        
        top_heads_df = pd.DataFrame({
            'rank': np.arange(1, len(layers) + 1),
            'layer': layers,
            'head': heads,
            'attention_score': motif_attention_heatmap[layers, heads]
        })
        top_heads_df.to_csv(os.path.join(output_dir, 'top_motif_attending_heads.csv'), index=False)
        
        plt.figure(figsize=(10, 5))
        bars = plt.bar(np.arange(len(layers)), motif_attention_heatmap[layers, heads], color=juelich_colors.custom_colors['raspberry'], alpha=0.8, edgecolor=juelich_colors.custom_colors['julich_blue_1'], linewidth=2)
        plt.xticks(np.arange(len(layers)), [f"L{l}H{h}" for l, h in zip(layers, heads)], rotation=45)
        plt.ylabel('Average Attention Score')
        plt.title('Top Heads Attending to RNA Motifs')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_motif_attending_heads.svg'), dpi=300, transparent=True, format='svg')
        plt.close()