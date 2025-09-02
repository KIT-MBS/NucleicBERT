import torch
import pytorch_lightning as pl
import torchmetrics as tm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import os

from nucleicbert.models.bert import BERT, NB_CONFIG
from nucleicbert.pretrain.utils import Config
from nucleicbert.pretrain.pretrainingdataset import RNASeqDataset 
from nucleicbert.utils.focal_loss import FocalLossMultiClass
from nucleicbert.downstream.closenessmodule import BERTWithClosenessResNet
from nucleicbert.downstream.closenessdataset import RNAClosenessDataset
from nucleicbert.downstream.secstrmodule import BERTWithSecStrLinear
from nucleicbert.downstream.secstrdataset import RNASecStrDataset
SAVE_DIR = 'saliency_outputs/'

def plot_saliency_with_structure_fixed(saliency_map, sequence, structure, save_path, input_ids=None, tokenizer=None):
    """
    Create enhanced saliency visualization with RNA secondary structure - FIXED VERSION
    
    Args:
        saliency_map (torch.Tensor): 1D saliency map tensor
        sequence (str): RNA sequence (AUGC)
        structure (str): Dot-bracket notation of RNA structure
        save_path (str): Path to save the visualization
        input_ids (torch.Tensor, optional): Input token IDs for context
        tokenizer (optional): Tokenizer to decode tokens
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Rectangle, Arc
    import matplotlib.colors as mcolors
    
    # Convert saliency map to numpy if it's a tensor
    if hasattr(saliency_map, 'detach'):
        saliency = saliency_map.squeeze().detach().cpu().numpy()
    else:
        saliency = np.array(saliency_map)
    
    # If saliency is 0 everywhere, we need to handle this case
    if np.max(saliency) == 0:
        saliency = np.ones_like(saliency) * 0.01  # Set to small value to avoid division by zero
    
    # Make sure sequence and structure are strings
    sequence = str(sequence)
    structure = str(structure)
    
    # Ensure sequence, structure and saliency have the same length
    min_len = min(len(sequence), len(structure), saliency.shape[-1])
    print(min_len)
    sequence = sequence[:min_len]
    structure = structure[:min_len]
    saliency = saliency[:min_len]
    
    # Debug information
    print(f"Debug - Lengths: sequence={len(sequence)}, structure={len(structure)}, saliency={saliency.shape[-1]}")
    print(f"Debug - Sample sequence: {sequence[:10]}...")
    print(f"Debug - Sample structure: {structure[:10]}...")
    print(f"Debug - Saliency range: min={np.min(saliency)}, max={np.max(saliency)}")
    
    # Parse the structure into base pairs
    stack = []
    base_pairs = []
    unpaired_positions = []
    
    for i, char in enumerate(structure):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                j = stack.pop()
                base_pairs.append((j, i))
        else:  # char == '.'
            unpaired_positions.append(i)
    
    # Create a structural feature vector (1 for paired, 0 for unpaired)
    structure_feature = np.zeros(len(structure))
    for i, j in base_pairs:
        structure_feature[i] = 1
        structure_feature[j] = 1
    
    # Normalize saliency for visualization
    if np.max(saliency) > 0:
        norm_saliency = saliency / np.max(saliency)
    else:
        norm_saliency = saliency
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(4, 1, height_ratios=[1, 1, 1, 2])
    
    # 1. Top plot: Saliency map
    ax1 = fig.add_subplot(gs[0])
    positions = np.arange(saliency.shape[-1])
    ax1.plot(positions, saliency, color='darkred', linewidth=1.2)
    ax1.set_title('Saliency Map')
    ax1.set_ylabel('Saliency')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, saliency.shape[-1])
    
    # Add x-axis ticks to show positions
    step = max(1, saliency.shape[-1] // 10)  # Show at most 10 position markers
    ax1.set_xticks(np.arange(0, saliency.shape[-1], step))
    
    # 2. Middle plot: Secondary structure
    ax2 = fig.add_subplot(gs[1])
    # Plot paired regions
    ax2.bar(positions, structure_feature, color='skyblue', alpha=0.7, width=1.0)
    ax2.set_title('RNA Secondary Structure (1 = paired, 0 = unpaired)')
    ax2.set_ylabel('Paired Status')
    ax2.set_xlim(0, saliency.shape[-1])
    ax2.set_ylim(0, 1.2)
    ax2.set_xticks(np.arange(0, saliency.shape[-1], step))
    
    # 3. Third plot: Correlation between saliency and structure
    ax3 = fig.add_subplot(gs[2])
    
    # Plot both on same axis
    ax3.plot(positions, norm_saliency, color='darkred', linewidth=1.2, label='Normalized Saliency')
    ax3.plot(positions, structure_feature, color='skyblue', linewidth=1.2, label='Structure (paired=1)')
    ax3.set_title('Saliency vs Structure Correlation')
    ax3.set_ylabel('Value')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, saliency.shape[-1])
    ax3.set_xticks(np.arange(0, saliency.shape[-1], step))
    
    # 4. Bottom plot: Combined visualization with sequence
    ax4 = fig.add_subplot(gs[3])
    
    # Create a custom colormap for saliency (white to red)
    cmap = plt.cm.Reds
    
    # Plot the sequence with coloring based on saliency
    for i, (nt, sal) in enumerate(zip(sequence, norm_saliency)):
        # Determine color based on saliency
        color = cmap(sal)
        
        # Add vertical bar for each nucleotide, height based on whether it's paired
        height = 0.8 if structure_feature[i] == 1 else 0.4
        ax4.add_patch(Rectangle((i-0.4, 0), 0.8, height, color=color, alpha=0.8))
        
        # Add nucleotide label
        ax4.text(i, height + 0.05, nt, ha='center', va='bottom', fontsize=8, 
                 fontweight='bold' if structure_feature[i] == 1 else 'normal')
    
    # Draw connecting arcs for base pairs
    for i, j in base_pairs:
        # Draw arc connecting the base pair
        # Calculate arc height based on distance
        distance = j - i
        arc_height = min(0.5, max(0.2, distance * 0.02))
        
        # Create the arc
        arc = Arc((i + (j-i)/2, 0.9), 
                  width=distance, 
                  height=arc_height*2, 
                  theta1=0, theta2=180, 
                  color='gray', alpha=0.5, linewidth=0.5)
        ax4.add_patch(arc)
    
    # Add a colorbar for saliency
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax4)
    cbar.set_label('Normalized Saliency')
    
    # Set axis limits for sequence view
    ax4.set_xlim(-1, saliency.shape[-1])
    ax4.set_ylim(0, 2.0)
    ax4.set_title('Sequence with Structure and Saliency')
    ax4.set_xlabel('Position')
    ax4.set_yticks([])
    ax4.set_xticks(np.arange(0, saliency.shape[-1], step))
    
    # Add statistical analysis
    paired_saliency = [saliency[i] for i, j in enumerate(structure_feature) if j == 1]
    unpaired_saliency = [saliency[i] for i, j in enumerate(structure_feature) if j == 0]
    
    mean_paired = np.mean(paired_saliency) if paired_saliency else 0
    mean_unpaired = np.mean(unpaired_saliency) if unpaired_saliency else 0
    
    stats_text = f"Mean saliency: Paired = {mean_paired:.4f}, Unpaired = {mean_unpaired:.4f}"
    fig.text(0.5, 0.01, stats_text, ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig

def analyze_structure_vs_saliency_fixed(saliency_map, sequence, structure, save_path):
    """
    Perform detailed statistical analysis of the relationship between
    RNA secondary structure and saliency values - FIXED VERSION
    
    Args:
        saliency_map (torch.Tensor or numpy.ndarray): 1D saliency map
        sequence (str): RNA sequence
        structure (str): Dot-bracket notation of RNA structure
        save_path (str): Path to save the analysis results
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.stats as stats
    from matplotlib.gridspec import GridSpec
    
    # Convert to numpy if needed
    print(saliency_map)
    if hasattr(saliency_map, 'detach'):
        saliency = saliency_map.squeeze().detach().cpu().numpy()
    else:
        saliency = np.array(saliency_map)
    
    # Debug information
    print(f"Statistical Analysis - Input lengths: sequence={len(sequence)}, structure={len(structure)}, saliency={saliency.shape[-1]}")
    
    # Ensure all inputs have the same length
    min_len = min(len(sequence), len(structure), saliency.shape[-1])
    sequence = sequence[:min_len]
    structure = structure[:min_len]
    saliency = saliency[:min_len]
    
    print(f"Statistical Analysis - Adjusted lengths: {min_len}")
    
    # Parse structure into features
    paired_positions = []
    unpaired_positions = []
    stack = []
    
    # First pass: identify paired positions
    for i, char in enumerate(structure):
        if char == '(':
            stack.append(i)
            paired_positions.append(i)
        elif char == ')':
            if stack:
                j = stack.pop()
                paired_positions.append(i)
        else:  # char == '.'
            unpaired_positions.append(i)
    
    # Create binary paired/unpaired vector
    structure_feature = np.zeros(len(structure))
    for i in paired_positions:
        structure_feature[i] = 1
    
    # Perform statistical analysis
    paired_saliency = saliency[structure_feature == 1]
    unpaired_saliency = saliency[structure_feature == 0]
    
    print(f"Statistical Analysis - Found {len(paired_saliency)} paired and {len(unpaired_saliency)} unpaired positions")
    
    # Descriptive statistics
    mean_paired = np.mean(paired_saliency) if len(paired_saliency) > 0 else 0
    mean_unpaired = np.mean(unpaired_saliency) if len(unpaired_saliency) > 0 else 0
    median_paired = np.median(paired_saliency) if len(paired_saliency) > 0 else 0
    median_unpaired = np.median(unpaired_saliency) if len(unpaired_saliency) > 0 else 0
    
    # Statistical tests
    u_stat, p_value = 0, 1
    correlation, corr_p_value = 0, 1
    
    if len(paired_saliency) > 0 and len(unpaired_saliency) > 0:
        # Mann-Whitney U test (non-parametric test for different distributions)
        try:
            u_stat, p_value = stats.mannwhitneyu(paired_saliency, unpaired_saliency, alternative='two-sided')
        except ValueError as e:
            print(f"Statistical test error: {e}")
            u_stat, p_value = 0, 1
        
        # Correlation between structure and saliency
        try:
            correlation, corr_p_value = stats.pearsonr(structure_feature, saliency)
        except ValueError as e:
            print(f"Correlation calculation error: {e}")
            correlation, corr_p_value = 0, 1
    
    # Create visualization
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2)
    
    # 1. Box plot comparing paired vs unpaired
    ax1 = fig.add_subplot(gs[0, 0])
    
    if len(paired_saliency) > 0 and len(unpaired_saliency) > 0:
        ax1.boxplot([paired_saliency, unpaired_saliency], labels=['Paired', 'Unpaired'])
        ax1.set_title('Saliency Distribution: Paired vs Unpaired Positions')
        ax1.set_ylabel('Saliency Value')
    else:
        ax1.text(0.5, 0.5, "Not enough data for boxplot", ha='center', va='center')
    
    # Add stats to plot
    stats_text = f"Mean: Paired={mean_paired:.4f}, Unpaired={mean_unpaired:.4f}\n"
    stats_text += f"Median: Paired={median_paired:.4f}, Unpaired={median_unpaired:.4f}\n"
    stats_text += f"Mann-Whitney p-value: {p_value:.4f}\n"
    stats_text += f"Correlation: {correlation:.4f} (p={corr_p_value:.4f})"
    
    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             verticalalignment='top')
    
    # 2. Histogram of saliency values for paired vs unpaired
    ax2 = fig.add_subplot(gs[0, 1])
    
    if len(paired_saliency) > 0 and len(unpaired_saliency) > 0:
        # Use fewer bins if small sample sizes
        bins = min(20, max(5, min(len(paired_saliency), len(unpaired_saliency)) // 2))
        
        ax2.hist(paired_saliency, bins=bins, alpha=0.5, label='Paired', color='blue')
        ax2.hist(unpaired_saliency, bins=bins, alpha=0.5, label='Unpaired', color='red')
        ax2.set_title('Saliency Value Distribution')
        ax2.set_xlabel('Saliency Value')
        ax2.set_ylabel('Frequency')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "Not enough data for histogram", ha='center', va='center')
    
    # 3. Scatter plot of structure vs saliency
    ax3 = fig.add_subplot(gs[1, 0])
    
    if len(structure_feature) > 0 and np.max(structure_feature) > 0:
        # Jitter the structure values slightly for better visualization
        jittered_structure = structure_feature + np.random.normal(0, 0.05, len(structure_feature))
        ax3.scatter(jittered_structure, saliency, alpha=0.5)
        ax3.set_title('Structure vs Saliency')
        ax3.set_xlabel('Structure (0=unpaired, 1=paired)')
        ax3.set_ylabel('Saliency Value')
        
        # Add trend line
        z = np.polyfit(structure_feature, saliency, 1)
        p = np.poly1d(z)
        ax3.plot([0, 1], [p(0), p(1)], "r--", alpha=0.8)
    else:
        ax3.text(0.5, 0.5, "No structure variation for scatter plot", ha='center', va='center')
    
    # 4. Plot average saliency per position type
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Define structural classes
    structural_types = ['Paired', 'Unpaired']
    saliency_means = [mean_paired, mean_unpaired]
    
    # Create bar chart
    bars = ax4.bar(structural_types, saliency_means, color=['blue', 'red'])
    ax4.set_title('Average Saliency by Structural Element')
    ax4.set_ylabel('Mean Saliency')
    
    # Add error bars (standard deviation)
    if len(paired_saliency) > 0:
        paired_std = np.std(paired_saliency)
    else:
        paired_std = 0
        
    if len(unpaired_saliency) > 0:
        unpaired_std = np.std(unpaired_saliency)
    else:
        unpaired_std = 0
        
    ax4.errorbar(x=[0, 1], y=saliency_means, yerr=[paired_std, unpaired_std], 
                fmt='none', ecolor='black', capsize=5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save detailed stats as text file
    text_path = save_path.replace('.png', '_stats.txt')
    with open(text_path, 'w') as f:
        f.write("RNA Structure vs Saliency Statistical Analysis\n")
        f.write("=============================================\n\n")
        f.write(f"Sequence length: {len(sequence)}\n")
        f.write(f"Paired positions: {len(paired_positions)}\n")
        f.write(f"Unpaired positions: {len(unpaired_positions)}\n\n")
        
        f.write("Descriptive Statistics:\n")
        f.write(f"Mean saliency (paired): {mean_paired:.6f}\n")
        f.write(f"Mean saliency (unpaired): {mean_unpaired:.6f}\n")
        f.write(f"Median saliency (paired): {median_paired:.6f}\n")
        f.write(f"Median saliency (unpaired): {median_unpaired:.6f}\n")
        f.write(f"Std dev (paired): {paired_std:.6f}\n")
        f.write(f"Std dev (unpaired): {unpaired_std:.6f}\n\n")
        
        f.write("Statistical Tests:\n")
        f.write(f"Mann-Whitney U test: U={u_stat:.6f}, p-value={p_value:.6f}\n")
        f.write(f"Pearson correlation: r={correlation:.6f}, p-value={corr_p_value:.6f}\n\n")
        
        if p_value < 0.05:
            f.write("SIGNIFICANT difference between paired and unpaired regions.\n")
        else:
            f.write("NO significant difference between paired and unpaired regions.\n")
    
    return {
        'mean_paired': mean_paired,
        'mean_unpaired': mean_unpaired,
        'p_value': p_value,
        'correlation': correlation
    }

def plot_improved_saliency(saliency_map, save_path, input_ids=None, tokenizer=None):
    """
    Create an improved saliency visualization, stopping at [SEP] token
    
    Args:
        saliency_map (torch.Tensor): 1D saliency map tensor
        input_ids (torch.Tensor, optional): Input token IDs for context
        tokenizer (optional): Tokenizer to decode tokens
        save_path (str): Path to save the visualization
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec
    
    # Convert to numpy
    saliency = saliency_map.detach().cpu().numpy()
    
    # If input_ids are provided, trim the sequence at [SEP] token
    if input_ids is not None:
        ids = input_ids.detach().cpu().numpy()
        
        # Find [SEP] token - depends on tokenizer implementation
        sep_id = None
        if hasattr(tokenizer, 'convert_tokens_to_ids'):
            sep_id = tokenizer.convert_tokens_to_ids('[SEP]')
        
        if sep_id is not None:
            # Find the first occurrence of SEP
            sep_indices = np.where(ids == sep_id)[0]
            print(f"SEP indices: {sep_indices}")
            if len(sep_indices) > 0:
                sep_idx = sep_indices[0]
                # Trim input_ids and saliency at SEP token and also remove CLS token
                ids = ids[1:sep_idx]  
                saliency = saliency[1:sep_idx]

    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 1, height_ratios=[1, 2, 2])
    
    # 1. Top plot: Line chart of the full sequence
    ax1 = fig.add_subplot(gs[0])
    positions = np.arange(saliency.shape[-1])
    ax1.plot(positions, saliency, color='darkred', linewidth=0.8)
    ax1.set_title('Full Sequence Saliency (Line Chart)')
    ax1.set_ylabel('Saliency')
    ax1.grid(True, alpha=0.3)
    
    # 2. Middle plot: Smoothed saliency
    ax2 = fig.add_subplot(gs[1])
    
    # Apply smoothing with a rolling window
    window_size = min(10, max(3, saliency.shape[-1] // 100))  # Adaptive window size
    smoothed_saliency = np.convolve(saliency, np.ones(window_size)/window_size, mode='same')
    
    ax2.plot(positions, smoothed_saliency, color='blue', linewidth=1.5)
    ax2.set_title(f'Smoothed Saliency (Window Size: {window_size})')
    ax2.set_ylabel('Smoothed Saliency')
    ax2.grid(True, alpha=0.3)
    
    # Find peaks (local maxima) in the smoothed saliency
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(smoothed_saliency, height=np.percentile(smoothed_saliency, 85), 
                          distance=max(5, saliency.shape[-1] // 50))
    
    # Highlight peaks
    ax2.scatter(peaks, smoothed_saliency[peaks], color='red', s=50, zorder=5)
    for peak in peaks:
        ax2.annotate(f"{peak}", (peak, smoothed_saliency[peak]), 
                     xytext=(0, 10), textcoords='offset points', 
                     ha='center', fontsize=8)
    
    # 3. Bottom plot: Top 10% most salient positions as a bar chart
    ax3 = fig.add_subplot(gs[2])
    
    # Get the indices of the top 10% most salient tokens
    top_k = max(1, int(saliency.shape[-1] * 0.1))  # At least 1 token
    top_indices = np.argsort(saliency)[-top_k:]
    top_saliency = saliency[top_indices]
    
    # Sort by position for better visualization
    sorted_idx = np.argsort(top_indices)
    top_indices = top_indices[sorted_idx]
    top_saliency = top_saliency[sorted_idx]
    
    # Create a bar chart of top salient positions
    bars = ax3.bar(top_indices, top_saliency, width=max(1, saliency.shape[-1] // 200), color='darkred')
    ax3.set_title(f'Top {top_k} Most Salient Positions')
    ax3.set_xlabel('Position')
    ax3.set_ylabel('Saliency')
    ax3.grid(True, axis='y', alpha=0.3)
    
    # Add position labels
    for i, (pos, val) in enumerate(zip(top_indices, top_saliency)):
        ax3.annotate(f"{pos}", (pos, val), 
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', fontsize=8)
    
    # Add token context if available
    if input_ids is not None and tokenizer is not None:
        # Add token context for peaks
        for i, peak in enumerate(peaks):
            # Get context around peak (5 tokens before and after)
            start = max(0, peak - 5)
            end = min(len(ids), peak + 6)
            
            context_ids = ids[start:end]
            context_positions = list(range(start, end))
            
            # Convert to tokens
            context_tokens = []
            for token_id in context_ids:
                if token_id == 0:  # Skip padding
                    token = "[PAD]"
                elif hasattr(tokenizer, 'convert_ids_to_tokens'):
                    token = tokenizer.convert_ids_to_tokens(int(token_id))
                elif hasattr(tokenizer, 'decode'):
                    token = tokenizer.decode([int(token_id)])
                else:
                    token = str(token_id)
                context_tokens.append(token)
            
            # Highlight the peak token
            peak_idx = context_positions.index(peak) if peak in context_positions else -1
            
            # Format the context string
            context_str = " ".join(context_tokens)
            if peak_idx >= 0:
                # Bold the peak token in the string
                parts = context_str.split()
                parts[peak_idx] = f"**{parts[peak_idx]}**"
                context_str = " ".join(parts)
            
            # Add context annotation
            y_pos = smoothed_saliency[peak] + (i % 3 + 1) * 0.02 * max(smoothed_saliency)
            ax2.annotate(f"Context: {context_str}", 
                         (peak, y_pos),
                         xytext=(20, 20), textcoords='offset points',
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'),
                         bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3),
                         fontsize=8, ha='left')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig

def analyze_long_range_dependencies(saliency_map, save_path, input_ids=None, tokenizer=None):
    """
    Analyze and visualize potential long-range dependencies in the saliency map,
    stopping at [SEP] token
    
    Args:
        saliency_map (torch.Tensor): 1D saliency map tensor
        input_ids (torch.Tensor, optional): Input token IDs for context
        tokenizer (optional): Tokenizer to decode tokens
        save_path (str): Path to save the visualization
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec
    from scipy.signal import find_peaks
    
    # Convert to numpy
    saliency = saliency_map.detach().cpu().numpy()
    
    # If input_ids are provided, trim the sequence at [SEP] token
    if input_ids is not None:
        ids = input_ids.detach().cpu().numpy()
        
        # Find [SEP] token - depends on tokenizer implementation
        sep_id = None
        if hasattr(tokenizer, 'convert_tokens_to_ids'):
            sep_id = tokenizer.convert_tokens_to_ids('[SEP]')
        
        if sep_id is not None:
            # Find the first occurrence of SEP
            sep_indices = np.where(ids == sep_id)[0]
            if len(sep_indices) > 0:
                sep_idx = sep_indices[0]
                # Trim input_ids and saliency at SEP token and also remove CLS token
                ids = ids[1:sep_idx] 
                saliency = saliency[1:sep_idx]
    else:
        ids = None
    
    seq_len = saliency.shape[-1]
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(4, 1, height_ratios=[1, 2, 1, 3])
    
    # 1. Top plot: Line chart of the full sequence
    ax1 = fig.add_subplot(gs[0])
    positions = np.arange(seq_len)
    ax1.plot(positions, saliency, color='darkred', linewidth=0.8)
    ax1.set_title('Full Sequence Saliency (Line Chart)')
    ax1.set_ylabel('Saliency')
    ax1.grid(True, alpha=0.3)
    
    # Add horizontal line at a threshold
    threshold = np.percentile(saliency, 90)  # 90th percentile
    ax1.axhline(y=threshold, color='blue', linestyle='--', alpha=0.7, 
                label=f'90th Percentile: {threshold:.4f}')
    ax1.legend()
    
    # 2. Second plot: Distance analysis of salient regions
    ax2 = fig.add_subplot(gs[1])
    
    # Find peaks above threshold
    min_distance = max(5, seq_len // 100)  # Adaptive distance based on sequence length
    peaks, _ = find_peaks(saliency, height=threshold, distance=min_distance)
    
    # If no peaks found, adjust threshold
    if len(peaks) < 2:
        threshold = np.percentile(saliency, 80)
        peaks, _ = find_peaks(saliency, height=threshold, distance=min_distance)
        ax2.set_title(f'No peaks found at 90th percentile. Adjusted to 80th: {threshold:.4f}')
    else:
        ax2.set_title(f'Salient Regions and Their Distances (Threshold: {threshold:.4f})')
    
    # Plot saliency peaks
    ax2.plot(positions, saliency, color='grey', alpha=0.5, linewidth=0.8)
    ax2.scatter(peaks, saliency[peaks], color='red', s=50, zorder=5, label='Peaks')
    
    # Analyze distances between peaks
    distances = []
    for i in range(len(peaks)-1):
        distances.append(peaks[i+1] - peaks[i])
        mid_point = (peaks[i] + peaks[i+1]) / 2
        height = max(saliency[peaks[i]], saliency[peaks[i+1]]) * 1.1
        
        # Draw connecting lines between peaks
        ax2.plot([peaks[i], peaks[i+1]], [saliency[peaks[i]], saliency[peaks[i+1]]], 
                 'b--', alpha=0.4)
        
        # Label the distance
        ax2.annotate(f"d={peaks[i+1]-peaks[i]}", 
                     (mid_point, height),
                     fontsize=8, ha='center')
    
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Saliency')
    ax2.legend()
    
    # 3. Third plot: Distribution of distances between salient regions
    ax3 = fig.add_subplot(gs[2])
    
    if distances:
        bins = max(5, min(20, len(distances)))
        ax3.hist(distances, bins=bins, color='skyblue', edgecolor='black')
        ax3.set_title('Distribution of Distances Between Salient Regions')
        ax3.set_xlabel('Distance')
        ax3.set_ylabel('Frequency')
        
        # Add statistics
        mean_dist = np.mean(distances)
        median_dist = np.median(distances)
        max_dist = np.max(distances)
        
        stats_text = f"Mean: {mean_dist:.1f}, Median: {median_dist:.1f}, Max: {max_dist:.1f}"
        ax3.annotate(stats_text, (0.5, 0.9), xycoords='axes fraction', ha='center',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))
        
        # Highlight long-range dependencies (distances above 75th percentile)
        long_range_threshold = np.percentile(distances, 75)
        ax3.axvline(x=long_range_threshold, color='red', linestyle='--', 
                   label=f'Long-range threshold: {long_range_threshold:.1f}')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, "Not enough peaks found to analyze distances", 
                ha='center', va='center', transform=ax3.transAxes)
    
    # 4. Fourth plot: Detailed view of salient tokens and their context
    ax4 = fig.add_subplot(gs[3])
    
    # Skip this plot if input_ids or tokenizer not provided
    if ids is None or tokenizer is None:
        ax4.text(0.5, 0.5, "Token IDs or tokenizer not provided - cannot show token context", 
                ha='center', va='center', transform=ax4.transAxes)
    else:
        # Find potential long-range pairs
        long_range_pairs = []
        
        for i in range(len(peaks)-1):
            distance = peaks[i+1] - peaks[i]
            if len(distances) > 0 and distance > np.percentile(distances, 75):
                long_range_pairs.append((peaks[i], peaks[i+1]))
        
        # If no long-range pairs, just show all pairs of peaks
        if not long_range_pairs and len(peaks) >= 2:
            # Take the pair with the greatest distance
            max_dist_idx = np.argmax(distances)
            long_range_pairs = [(peaks[max_dist_idx], peaks[max_dist_idx+1])]
        
        # Initialize plot
        ax4.set_xlim(0, seq_len)
        ax4.set_ylim(0, len(long_range_pairs) + 1)
        ax4.set_title("Long-Range Dependencies Between Salient Regions")
        ax4.set_xlabel("Sequence Position")
        ax4.set_yticks([])
        
        if not long_range_pairs:
            ax4.text(0.5, 0.5, "No significant long-range dependencies found", 
                    ha='center', va='center', transform=ax4.transAxes)
        else:
            # For each long-range pair, show the tokens at both locations
            for i, (pos1, pos2) in enumerate(long_range_pairs):
                # Get context for first position
                start1 = max(0, pos1 - 5)
                end1 = min(len(ids), pos1 + 6)
                context_ids1 = ids[start1:end1]
                
                # Get context for second position
                start2 = max(0, pos2 - 5)
                end2 = min(len(ids), pos2 + 6)
                context_ids2 = ids[start2:end2]
                
                # Convert to tokens
                def get_token_text(token_ids):
                    tokens = []
                    for token_id in token_ids:
                        if token_id == 0:  # Skip padding
                            token = "[PAD]"
                        elif hasattr(tokenizer, 'convert_ids_to_tokens'):
                            token = tokenizer.convert_ids_to_tokens(int(token_id))
                        elif hasattr(tokenizer, 'decode'):
                            token = tokenizer.decode([int(token_id)])
                        else:
                            token = str(token_id)
                        tokens.append(token)
                    return tokens
                
                context_tokens1 = get_token_text(context_ids1)
                context_tokens2 = get_token_text(context_ids2)
                
                # Find the index of the peak token in each context
                peak1_idx = pos1 - start1
                peak2_idx = pos2 - start2
                
                # Plot the relation
                y_pos = i + 0.5
                
                # Draw the connecting line
                ax4.plot([pos1, pos2], [y_pos, y_pos], 'b-', alpha=0.5)
                
                # Add points for the peaks
                ax4.scatter([pos1, pos2], [y_pos, y_pos], color='red', s=50, zorder=5)
                
                # Add distance label
                mid_point = (pos1 + pos2) / 2
                ax4.annotate(f"Distance: {pos2-pos1}", 
                            (mid_point, y_pos + 0.1),
                            ha='center', va='bottom', fontsize=9)
                
                # Add context boxes
                # First context
                context_str1 = " ".join(context_tokens1)
                if 0 <= peak1_idx < len(context_tokens1):
                    # Highlight the peak token
                    tokens = context_tokens1.copy()
                    tokens[peak1_idx] = f"*{tokens[peak1_idx]}*"
                    context_str1 = " ".join(tokens)
                
                ax4.annotate(f"Context: {context_str1}", 
                            (pos1, y_pos - 0.2),
                            xytext=(0, -15), textcoords='offset points',
                            ha='center', va='top', fontsize=8,
                            bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.3))
                
                # Second context
                context_str2 = " ".join(context_tokens2)
                if 0 <= peak2_idx < len(context_tokens2):
                    # Highlight the peak token
                    tokens = context_tokens2.copy()
                    tokens[peak2_idx] = f"*{tokens[peak2_idx]}*"
                    context_str2 = " ".join(tokens)
                
                ax4.annotate(f"Context: {context_str2}", 
                            (pos2, y_pos - 0.2),
                            xytext=(0, -15), textcoords='offset points',
                            ha='center', va='top', fontsize=8,
                            bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig

class SaliencyForward:
    def __init__(
            self,
            model,
            model_name,
            device='cuda',
        ):
        """
        Initialize the saliency forward pass class
        """
        self.model = model
        self.model_name = model_name
        self.device = device
        if model_name == 'bert':
            self.forward_fn = self.bert_saliency_forward
            self.loss_fn = torch.nn.CrossEntropyLoss(
                ignore_index=-100,
                reduction='mean',
            )
        elif model_name == 'bert_with_secstr':
            self.forward_fn = self.bert_saliency_forward_with_secstr
            self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)

        elif model_name == 'bert_with_closeness':
            self.forward_fn = self.bert_saliency_forward_with_closeness
            self.loss_fn = FocalLossMultiClass(
                alpha=0.8,
                gamma=2,
                ignore_index=-1,
            )
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        
    def bert_saliency_forward(
            self,
            input_ids,
            target_ids,
        ):

        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)

        input_embd = self.model.encoder.embedding.token(input_ids)
        
        self.model.zero_grad()
        
        position_embd = self.model.encoder.embedding.position(input_embd)
        input_embd = input_embd + position_embd

        hidden_states = self.model.encoder.embedding.dropout(input_embd)

        attention_mask = input_ids.ne(0).unsqueeze(2)
        attention_mask = attention_mask.to(dtype=torch.float32)
        
        encoder_output = self.model.encoder.encoder_forward(hidden_states, attention_mask, need_weights = False)
        
        outputs = self.model.mlm(encoder_output[0])
        
        loss = self.loss_fn(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
        input_embd.retain_grad()
        loss.backward()
        saliency_map = input_embd.grad.abs().mean(dim=-1)
        max_val = saliency_map.max()
        if max_val > 0:
            saliency_map = saliency_map / max_val

        return saliency_map

    def bert_saliency_forward_with_secstr(
            self,
            input_ids,
            target_ids,
        ):
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)
        input_embd = self.model.bert.encoder.embedding.token(input_ids)
        
        self.model.zero_grad()
        
        position_embd = self.model.bert.encoder.embedding.position(input_embd)
        input_embd = input_embd + position_embd

        hidden_states = self.model.bert.encoder.embedding.dropout(input_embd)

        attention_mask = input_ids.ne(0).unsqueeze(2)
        attention_mask = attention_mask.to(dtype=torch.float32)
        
        encoder_output = self.model.bert.encoder.encoder_forward(hidden_states, attention_mask, need_weights = True)
        input_list = encoder_output[2] # list of [B, 1, L, Hidden_Dim]
        embeddings = torch.cat(input_list, dim=1) # [B, Hidden_Layers+1, L, Hidden_Dim]
        embeddings = torch.mean(embeddings, dim=1) # [B, seq_len, Hidden_Dim]
        outputs = self.model.linear(embeddings)
        
        loss = self.loss_fn(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
        input_embd.retain_grad()
        loss.backward()
        saliency_map = input_embd.grad.abs().mean(dim=-1)
        max_val = saliency_map.max()
        if max_val > 0:
            saliency_map = saliency_map / max_val

        return saliency_map
    
    def bert_saliency_forward_with_closeness(
            self,
            input_ids,
            target_ids,
        ):
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)
        input_embd = self.model.bert.encoder.embedding.token(input_ids)

        self.model.zero_grad()

        position_embd = self.model.bert.encoder.embedding.position(input_embd)
        input_embd = input_embd + position_embd

        hidden_states = self.model.bert.encoder.embedding.dropout(input_embd)

        attention_mask = input_ids.ne(0).unsqueeze(2)
        attention_mask = attention_mask.to(dtype=torch.float32)

        encoder_output = self.model.bert.encoder.encoder_forward(hidden_states, attention_mask, need_weights = True)
        input_list = encoder_output[1] # list of [B, 1, H, L, L]
        attn_weights = torch.cat(input_list, dim=1) # [B, Layers, H, L, L]
        attn_weights = attn_weights[..., :-1, :-1] # remove the SEP token
        attn_weights = attn_weights[..., 1:, 1:] # remove the CLS token

        outputs = self.model.resnet(attn_weights)

        loss = self.loss_fn(outputs, target_ids)
        input_embd.retain_grad()
        loss.backward()
        saliency_map = input_embd.grad.abs().mean(dim=-1)
        max_val = saliency_map.max()
        if max_val > 0:
            saliency_map = saliency_map / max_val

        return saliency_map

class SaliencyMapGenerator:
    def __init__(
            self, 
            model_path,
            model_name,
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
        
        self.model = self._load_model(model_path, model_name)
        self.saliency_forward = SaliencyForward(
            self.model,
            model_name,
            device=device,
        )


    def _load_model(self, model_path, model_name):
        """
        Load the pre-trained model from a checkpoint
        
        Args:
            model_path (str): Path to the model checkpoint
        
        Returns:
            PreTrainingModule: Loaded model
        """
        if model_name == 'bert':
            model = BERT(**self.model_config)
            if self.model_path is not None:
                model.load_state_dict(torch.load(model_path))
        elif model_name == 'bert_with_secstr':
            bert = BERT(**NB_CONFIG)
            model = BERTWithSecStrLinear(
                bert,
                hidden_size=NB_CONFIG['hidden_size'],
                num_classes=15,
            )
            if self.model_path is not None:
                model.load_state_dict(torch.load(model_path))

        elif model_name == 'bert_with_closeness':
            bert = BERT(**NB_CONFIG)
            model = BERTWithClosenessResNet(
                bert,
                input_channels=NB_CONFIG['num_attention_heads']*NB_CONFIG['num_hidden_layers'],
                num_residual_blocks=1,
                task='contact_map'
            )
            if self.model_path is not None:
                model.load_state_dict(torch.load(model_path))
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        model = model.to(self.device)
        model.eval()  # Set to evaluation mode
        return model

    def generate_salient_maps(self, data_loader, save_dir):
        """
        Generate saliency maps for the input data
        
        Args:
            data_loader (DataLoader): DataLoader for the input data
            save_dir (str): Directory to save the saliency maps
        """
        os.makedirs(save_dir, exist_ok=True)
        for i, batch in enumerate(tqdm.tqdm(data_loader)):
            input_ids, target_ids = batch[0], batch[1]
            saliency_map = self.saliency_forward.forward_fn(input_ids, target_ids)
            print(saliency_map.shape)
            
            
            # Create visualizations for each sequence in the batch
            for j in range(saliency_map.size(0)):
                print(input_ids[j].shape)
                # 1. Basic heatmap visualization (correctly handles 1D tensor)
                plot_improved_saliency(
                    saliency_map[j], 
                    os.path.join(save_dir, f'saliency_improved_{i}_{j}_test.png'),
                    input_ids=input_ids[j],
                    tokenizer=self.tokenizer
                )
                analyze_long_range_dependencies(
                    saliency_map[j], 
                    os.path.join(save_dir, f'long_range_{i}_{j}_test.png'),
                    input_ids=input_ids[j],
                    tokenizer=self.tokenizer
                )
        
        return saliency_map

    


if __name__ == '__main__':
    from transformers import PreTrainedTokenizerFast
    pl.seed_everything(42, workers=True)
    model_path = '../logs/lightning_logs/Faster_48_2/experiment_h1024_l1024_h32_l32/state_dicts/model_state_dict_epoch=295-Validation Loss=0.345.pth'
    # model_path = None
    
    tokenizer = PreTrainedTokenizerFast(tokenizer_file='nucleicbert/tokenizers/noncoding_seqs.json')
    saliency_map_generator = SaliencyMapGenerator(
        model_path,
        tokenizer=tokenizer,
        device='cuda',
        model_name = 'bert',
    )

    # -----------------------------------------------This is for saliency maps without structure, only on the pretrained BERT-------
    val_dataset_indices = torch.load('val_dataset_indices.pt')[:10]
    dataset = RNASeqDataset(
        input = '../mars_data/processed_data_python/noncoding_seqs',
        tokenizer = tokenizer,
        constant_mask_positions = None,
        max_length = 1024,
        mask_lm_prob = 0.15,
    )
    val_dataset = torch.utils.data.Subset(
        dataset,
        val_dataset_indices,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
    )
    # -----------------------------------------------This is for saliency maps without structure, only on the pretrained BERT-------

    # -----------------------------------------------This is for saliency maps with structure---------------------------------------
    # sequences = ['AUAUCGAAAGGGCAAACCUGUCGAAAGGCAGGGGCGCAAAGCCAUGGGCCUGUCGGAAGUAAAACUUCCUAUGGUUGCCAGGCUGCCGAA']
    # sec_strs = ['...........((...((((((....))))))...))..((((.((((((....((((.......))))...)))..)))))))......']
    # input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sequences[0]))).unsqueeze(0)
    # target_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sequences[0]))).unsqueeze(0)
    # dataset = torch.utils.data.TensorDataset(input_ids, target_ids)
    # val_dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=1,
    #     shuffle=False,
    # )

    # secstr_dataset = RNASecStrDataset(
    #     data_dir = '../data/sec_str_data/bpRNA_80/sec_str_ts0.csv',
    #     tokenizer = tokenizer,
    #     max_length = 1024,
    #     min_length = 20,
    # )
    # secstr_dataset = torch.utils.data.Subset(
    #     secstr_dataset,
    #     range(1),
    # )
    # val_dataloader = torch.utils.data.DataLoader(
    #     secstr_dataset,
    #     batch_size=1,
    #     shuffle=False,
    # )
    # sequence = secstr_dataset[0][-2]
    # sec_str = secstr_dataset[0][-1]
    # ------------------------------------------------This is for saliency maps with structure---------------------------------------

    # ------------------------------------------------This is for saliency maps with closeness---------------------------------------
    # closeness_dataset = RNAClosenessDataset(
    #     input_seq = '../data/ns_bgsu_contact_map/val/inputs/',
    #     target = '../data/ns_bgsu_contact_map/val/targets/',
    #     tokenizer = tokenizer,
    # )
    # val_dataloader = torch.utils.data.DataLoader(
    #     closeness_dataset,
    #     batch_size=1,
    #     shuffle=False,
    # )

    # ------------------------------------------------This is for saliency maps with closeness---------------------------------------

    # print(sequence, sec_str)
    saliency_map = saliency_map_generator.generate_salient_maps(val_dataloader, SAVE_DIR)
    # # plot_saliency_with_structure_fixed(
    # #     saliency_map,
    # #     sequences[0],
    # #     sec_strs[0],
    # #     os.path.join(SAVE_DIR, 'saliency_structure.png')
    # # )

    # analyze_structure_vs_saliency_fixed(
    #     saliency_map,
    #     sequence,
    #     sec_str,
    #     os.path.join(SAVE_DIR, 'structure_vs_saliency.png')
    # )