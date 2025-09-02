import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import matplotlib
from matplotlib.gridspec import GridSpec

import juelich_colors as jc

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

def plot_saliency_contactmap_correlation(
    saliency_map: torch.Tensor,
    contact_map: torch.Tensor,
    save_path: str,
    input_ids: Optional[torch.Tensor] = None,
    tokenizer = None,
    figsize=(14, 3),
    palette_type='primary'
) -> plt.Figure:
    """
    Analyze the correlation between saliency and contact density - focused view
    
    Args:
        saliency_map: Saliency map tensor [seq_len]
        contact_map: Contact map tensor [seq_len, seq_len]
        save_path: Path to save the visualization
        input_ids: Input token IDs for context
        tokenizer: Tokenizer to decode token IDs
        figsize: Figure size (width, height)
        palette_type: Color palette type ('primary', 'secondary', 'all')
        
    Returns:
        Matplotlib figure and correlation coefficient
    """
    
    jc.set_matplotlib_color_cycle(palette_type)
    
    if hasattr(saliency_map, 'detach'):
        saliency = saliency_map.squeeze().detach().cpu().numpy()
    else:
        saliency = np.array(saliency_map).squeeze()
    
    if hasattr(contact_map, 'detach'):
        contact = contact_map.squeeze().detach().cpu().numpy()
    else:
        contact = np.array(contact_map).squeeze()
    
    if input_ids is not None and tokenizer is not None:
        if hasattr(input_ids, 'detach'):
            ids = input_ids.detach().cpu().numpy()
        else:
            ids = np.array(input_ids)
        

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
    

    seq_len = len(saliency)
    if contact.shape[0] > seq_len:
        contact = contact[:seq_len, :seq_len]
    elif contact.shape[0] < seq_len:

        pad_size = seq_len - contact.shape[0]
        contact = np.pad(contact, ((0, pad_size), (0, pad_size)), 'constant')
    

    contact_density = np.sum(contact, axis=1) / seq_len
    

    correlation = np.corrcoef(saliency, contact_density)[0, 1]
    p_value = 0
    

    try:
        from scipy import stats
        _, p_value = stats.pearsonr(saliency, contact_density)
    except ImportError:
        pass
    

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 2, width_ratios=[1, 1])
    

    ax1 = fig.add_subplot(gs[0, 0])
    

    if np.max(saliency) > np.min(saliency):
        norm_saliency = (saliency - np.min(saliency)) / (np.max(saliency) - np.min(saliency))
    else:
        norm_saliency = saliency
        
    if np.max(contact_density) > np.min(contact_density):
        norm_density = (contact_density - np.min(contact_density)) / (np.max(contact_density) - np.min(contact_density))
    else:
        norm_density = contact_density
    

    positions = np.arange(seq_len)
    saliency_line = ax1.plot(positions, norm_saliency, 
                            color=jc.custom_colors['julich_blue_1'], 
                            linewidth=2.5, 
                            label='Normalized Saliency', 
                            alpha=1.0)
    
    density_line = ax1.plot(positions, norm_density, 
                           color=jc.custom_colors['raspberry'], 
                           linewidth=2.5, 
                           label='Normalized Contact Density', 
                           alpha=1.0)
    

    ax1.fill_between(positions, norm_saliency, alpha=0.2, 
                    color=jc.custom_colors['julich_blue_2'])
    ax1.fill_between(positions, norm_density, alpha=0.2, 
                    color=jc.custom_colors['raspberry'])
    

    ax1.set_title('Normalized Comparison', loc='left')
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Normalized Value')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.set_axisbelow(True)
    ax1.set_xlim(0, seq_len)
    ax1.set_ylim(0, 1.1)
    ax1.margins(x=0.01, y=0.05)
    
    ax2 = fig.add_subplot(gs[0, 1])
    
    scatter = ax2.scatter(contact_density, saliency, 
                         color=jc.custom_colors['raspberry'], 
                         alpha=1.0, s=200, edgecolors='white', linewidth=0.5)
    
    if len(contact_density) > 1 and np.std(contact_density) > 0:
        z = np.polyfit(contact_density, saliency, 1)
        p = np.poly1d(z)
        trend_x = np.linspace(np.min(contact_density), np.max(contact_density), 100)
        ax2.plot(trend_x, p(trend_x), 
                color=jc.custom_colors['julich_blue_2'], 
                linestyle='--', linewidth=2, alpha=1.0, label='Trend Line')
    
    ax2.set_title(f'Correlation Analysis\nr = {correlation:.3f}', loc='left')
    ax2.set_xlabel('Contact Density')
    ax2.set_ylabel('Saliency')
    ax2.set_axisbelow(True)
    ax2.margins(x=0.05, y=0.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, transparent=True, format='svg')
    plt.close()
    
    return fig


def analyze_contactmap_vs_saliency(
    saliency_map: torch.Tensor,
    contact_map: torch.Tensor,
    save_path: str,
    input_ids: Optional[torch.Tensor] = None,
    tokenizer = None
):
    """
    Perform detailed statistical analysis of the relationship between
    RNA contact map properties and saliency values
    
    Args:
        saliency_map: Saliency map tensor [seq_len]
        contact_map: Contact map tensor [seq_len, seq_len]
        save_path: Path to save the analysis results
        input_ids: Input token IDs for context
        tokenizer: Tokenizer to decode token IDs
        
    Returns:
        Dictionary containing statistical results
    """

    if hasattr(saliency_map, 'detach'):
        saliency = saliency_map.squeeze().detach().cpu().numpy()
    else:
        saliency = np.array(saliency_map).squeeze()
    
    if hasattr(contact_map, 'detach'):
        contact = contact_map.squeeze().detach().cpu().numpy()
    else:
        contact = np.array(contact_map).squeeze()
    
    if input_ids is not None and tokenizer is not None:
        if hasattr(input_ids, 'detach'):
            ids = input_ids.detach().cpu().numpy()
        else:
            ids = np.array(input_ids)
        
        # Find [SEP] token
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
    
    seq_len = len(saliency)
    if contact.shape[0] > seq_len:
        contact = contact[:seq_len, :seq_len]
    elif contact.shape[0] < seq_len:
        pad_size = seq_len - contact.shape[0]
        contact = np.pad(contact, ((0, pad_size), (0, pad_size)), 'constant')
    
    contact_features = {
        'degree': np.sum(contact, axis=1),  # Number of contacts (already binary)
        'avg_contact_prob': np.mean(contact, axis=1),  # Average contact probability
        'max_contact_prob': np.max(contact, axis=1),  # Maximum contact probability
        'contact_entropy': np.zeros(seq_len)  # Contact probability entropy
    }
    
    for i in range(seq_len):
        probs = contact[i] / np.sum(contact[i]) if np.sum(contact[i]) > 0 else np.zeros(seq_len)
        valid_probs = probs[probs > 0]
        if len(valid_probs) > 0:
            contact_features['contact_entropy'][i] = -np.sum(valid_probs * np.log2(valid_probs))
    
    long_range_threshold = 24  # Positions more than this apart are considered long-range
    long_range_contacts = np.zeros(seq_len)
    short_range_contacts = np.zeros(seq_len)
    
    for i in range(seq_len):
        for j in range(seq_len):
            if contact[i, j] == 1:  # Binary contact
                if abs(i - j) > long_range_threshold:
                    long_range_contacts[i] += 1
                else:
                    short_range_contacts[i] += 1
    
    contact_features['long_range_ratio'] = np.zeros(seq_len)
    has_contacts = (short_range_contacts + long_range_contacts) > 0
    contact_features['long_range_ratio'][has_contacts] = long_range_contacts[has_contacts] / (short_range_contacts[has_contacts] + long_range_contacts[has_contacts])
    

    correlations = {}
    p_values = {}
    
    try:
        from scipy import stats
        for feature_name, feature_values in contact_features.items():
            corr, p_val = stats.pearsonr(saliency, feature_values)
            correlations[feature_name] = corr
            p_values[feature_name] = p_val
    except ImportError:
        for feature_name, feature_values in contact_features.items():
            corr = np.corrcoef(saliency, feature_values)[0, 1]
            correlations[feature_name] = corr
            p_values[feature_name] = 0.0  # Cannot calculate p-value
    
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3)
    
    # 1. Top left: Bar chart of correlations
    ax1 = fig.add_subplot(gs[0, 0])
    feature_names = list(correlations.keys())
    correlation_values = [correlations[name] for name in feature_names]
    bars = ax1.bar(feature_names, correlation_values, color='skyblue')
    
    for i, (name, bar) in enumerate(zip(feature_names, bars)):
        if p_values[name] < 0.05:
            # Significant correlations
            if correlation_values[i] > 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
    
    ax1.set_title('Correlation between Saliency and Contact Features', fontsize=14)
    ax1.set_ylabel('Pearson Correlation', fontsize=12)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.set_ylim(-1, 1)
    
    for i, name in enumerate(feature_names):
        if p_values[name] < 0.001:
            stars = '***'
        elif p_values[name] < 0.01:
            stars = '**'
        elif p_values[name] < 0.05:
            stars = '*'
        else:
            stars = ''
            
        if stars:
            ax1.text(i, correlation_values[i] + 0.05 * np.sign(correlation_values[i]), 
                    stars, ha='center', fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    
    # 2. Top right: Scatter plot of the strongest correlation
    ax2 = fig.add_subplot(gs[0, 1])
    
    strongest_feature = feature_names[np.argmax(np.abs(correlation_values))]
    strongest_corr = correlations[strongest_feature]
    strongest_p = p_values[strongest_feature]
    
    ax2.scatter(contact_features[strongest_feature], saliency, alpha=0.6, color='purple')
    ax2.set_title(f'Strongest Correlation: {strongest_feature}\nr={strongest_corr:.3f}, p={strongest_p:.3e}', fontsize=14)
    ax2.set_xlabel(strongest_feature.replace('_', ' ').title(), fontsize=12)
    ax2.set_ylabel('Saliency', fontsize=12)
    ax2.grid(True, alpha=0.3)
    

    if len(contact_features[strongest_feature]) > 1:
        z = np.polyfit(contact_features[strongest_feature], saliency, 1)
        p = np.poly1d(z)
        x_range = np.linspace(min(contact_features[strongest_feature]), 
                             max(contact_features[strongest_feature]), 100)
        ax2.plot(x_range, p(x_range), "r--", alpha=0.8)
    
    # 3. Middle left: Contact degree vs saliency
    ax3 = fig.add_subplot(gs[1, 0])
    contact_degree = contact_features['degree']
    
    norm_saliency = (saliency - np.min(saliency)) / (np.max(saliency) - np.min(saliency)) if np.max(saliency) > np.min(saliency) else saliency
    norm_degree = (contact_degree - np.min(contact_degree)) / (np.max(contact_degree) - np.min(contact_degree)) if np.max(contact_degree) > np.min(contact_degree) else contact_degree
    
    ax3.plot(np.arange(seq_len), norm_saliency, color='darkred', linewidth=1.5, label='Normalized Saliency')
    ax3.plot(np.arange(seq_len), norm_degree, color='darkblue', linewidth=1.5, label='Normalized Contact Degree')
    ax3.set_title('Saliency vs Contact Degree', fontsize=14)
    ax3.set_xlabel('Sequence Position', fontsize=12)
    ax3.set_ylabel('Normalized Value', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(gs[1, 1])
    long_range_ratio = contact_features['long_range_ratio']
    
    norm_long_range = (long_range_ratio - np.min(long_range_ratio)) / (np.max(long_range_ratio) - np.min(long_range_ratio)) if np.max(long_range_ratio) > np.min(long_range_ratio) else long_range_ratio
    
    ax4.plot(np.arange(seq_len), norm_saliency, color='darkred', linewidth=1.5, label='Normalized Saliency')
    ax4.plot(np.arange(seq_len), norm_long_range, color='green', linewidth=1.5, label='Normalized Long-Range Ratio')
    ax4.set_title('Saliency vs Long-Range Contact Ratio', fontsize=14)
    ax4.set_xlabel('Sequence Position', fontsize=12)
    ax4.set_ylabel('Normalized Value', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Bottom: Heatmap showing all correlations
    ax5 = fig.add_subplot(gs[2, :])
    
    corr_matrix = np.zeros((6, 6))
    labels = list(contact_features.keys()) + ['saliency']
    feature_values = list(contact_features.values()) + [saliency]
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            try:
                corr_matrix[i, j] = np.corrcoef(feature_values[i], feature_values[j])[0, 1]
            except:
                corr_matrix[i, j] = 0
    
    im = ax5.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    
    ax5.set_xticks(np.arange(len(labels)))
    ax5.set_yticks(np.arange(len(labels)))
    ax5.set_xticklabels([l.replace('_', ' ').title() for l in labels], rotation=45, ha='right')
    ax5.set_yticklabels([l.replace('_', ' ').title() for l in labels])
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            text_color = 'white' if abs(corr_matrix[i, j]) > 0.5 else 'black'
            ax5.text(j, i, f"{corr_matrix[i, j]:.2f}", 
                    ha="center", va="center", color=text_color, fontsize=10)
    
    ax5.set_title('Correlation Matrix of All Features', fontsize=14)
    plt.colorbar(im, ax=ax5, label='Correlation')

    # 6. NEW: Bottom left: Contact map visualization
    ax6 = fig.add_subplot(gs[2, 0])
    
    contact_cmap = plt.cm.get_cmap('Blues')
    
    contact_im = ax6.imshow(contact, cmap=contact_cmap, interpolation='nearest')
    ax6.set_title('Contact Map', fontsize=14)
    ax6.set_xlabel('Sequence Position', fontsize=12)
    ax6.set_ylabel('Sequence Position', fontsize=12)
    
    plt.colorbar(contact_im, ax=ax6, label='Contact (0/1)')
    
    top_salient_threshold = np.percentile(saliency, 90)  # Top 10% salient positions
    top_salient_indices = np.where(saliency >= top_salient_threshold)[0]
    
    for idx in top_salient_indices:
        if idx < contact.shape[0]:
            ax6.add_patch(plt.Rectangle((0, idx - 0.5), contact.shape[1], 1, 
                                       fill=True, color='red', alpha=0.1))
            ax6.add_patch(plt.Rectangle((idx - 0.5, 0), 1, contact.shape[0], 
                                       fill=True, color='red', alpha=0.1))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, transparent=True, format='svg')
    plt.close()
    
    text_path = save_path.replace('.svg', '_summary.txt')
    with open(text_path, 'w') as f:
        f.write("Contact Map vs Saliency Statistical Analysis\n")
        f.write("==========================================\n\n")
        f.write(f"Sequence length: {seq_len}\n\n")
        
        f.write("Correlations with Saliency:\n")
        for feature, corr in correlations.items():
            p_val = p_values[feature]
            sig = ""
            if p_val < 0.05:
                sig = "*"
            if p_val < 0.01:
                sig = "**"
            if p_val < 0.001:
                sig = "***"
            f.write(f"{feature}: r = {corr:.4f} (p = {p_val:.4e}) {sig}\n")
        
        f.write("\nSummary:\n")
        if any(p < 0.05 for p in p_values.values()):
            f.write("Significant correlations found between saliency and contact features.\n")
            most_sig_feature = feature_names[np.argmin(list(p_values.values()))]
            f.write(f"The most significant relationship is with {most_sig_feature}.\n")
        else:
            f.write("No significant correlations found between saliency and contact features.\n")
    
    return {
        'correlations': correlations,
        'p_values': p_values,
        'strongest_feature': strongest_feature,
        'strongest_correlation': strongest_corr
    }