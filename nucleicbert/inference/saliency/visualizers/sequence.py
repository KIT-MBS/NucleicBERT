import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import matplotlib   

import juelich_colors as jc

def set_style():
    """Keep original function name but adjust for publication"""
    font = {
        'family': 'sans-serif',
        'size': 10  
    }
    axes = {
        'titlesize': 11,  
        'labelsize': 10,  
    }
    xtick = {
        'labelsize': 9,  
    }
    ytick = {
        'labelsize': 9,  
    }
    legend = {
        'fontsize': 8,  
        'title_fontsize': 8,  
        'markerscale': 1,  
    }
    lines = {
        'markersize': 4,  
    }
    
    matplotlib.rc('font', **font)
    matplotlib.rc('axes', **axes)
    matplotlib.rc('xtick', **xtick)
    matplotlib.rc('ytick', **ytick)
    matplotlib.rc('legend', **legend)
    matplotlib.rc('lines', **lines)


def plot_improved_saliency(saliency_map, save_path, input_ids=None, tokenizer=None, 
                          figsize=(7, 3), window_size=None, peak_threshold=85,
                          show_context=True, context_window=5, palette_type='primary'):
    """
    Create an improved smoothed saliency visualization using custom color palette
    
    Args:
        saliency_map (torch.Tensor): 1D saliency map tensor
        save_path (str): Path to save the visualization
        input_ids (torch.Tensor, optional): Input token IDs for context
        tokenizer (optional): Tokenizer to decode tokens
        figsize (tuple): Figure size (width, height) - REDUCED DEFAULT
        window_size (int, optional): Smoothing window size (auto-calculated if None)
        peak_threshold (int): Percentile threshold for peak detection (0-100)
        show_context (bool): Whether to show token context for peaks
        context_window (int): Number of tokens before/after peak to show in context
        palette_type (str): Color palette type ('primary', 'secondary', 'all')
    """
    
    jc.set_matplotlib_color_cycle(palette_type)
    
    saliency = saliency_map.detach().cpu().numpy()
    
    ids = None
    if input_ids is not None:
        ids = input_ids.detach().cpu().numpy()
        
        sep_id = None
        if hasattr(tokenizer, 'convert_tokens_to_ids'):
            sep_id = tokenizer.convert_tokens_to_ids('[SEP]')
        
        if sep_id is not None:
            sep_indices = np.where(ids == sep_id)[0]
            if len(sep_indices) > 0:
                sep_idx = sep_indices[0]
                # Trim input_ids and saliency at SEP token, remove CLS token
                ids = ids[1:sep_idx]  
                saliency = saliency[1:sep_idx]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    positions = np.arange(len(saliency))
    
    if window_size is None:
        window_size = min(10, max(3, len(saliency) // 100))
    
    if window_size > 1:
        weights = np.exp(-0.5 * ((np.arange(window_size) - window_size//2) / (window_size/6))**2)
        weights = weights / weights.sum()
        smoothed_saliency = np.convolve(saliency, weights, mode='same')
    else:
        smoothed_saliency = saliency.copy()
    
    line = ax.plot(positions, smoothed_saliency, color=jc.custom_colors['julich_blue_1'], 
                   linewidth=1.5, label=f'Smoothed Saliency (Window: {window_size})', alpha=0.9)
    
    ax.fill_between(positions, smoothed_saliency, alpha=0.2, color=jc.custom_colors['julich_blue_2'])
    
    peak_height = np.percentile(smoothed_saliency, peak_threshold)
    min_distance = max(5, len(saliency) // 50)
    peaks, peak_properties = find_peaks(smoothed_saliency, 
                                       height=peak_height, 
                                       distance=min_distance,
                                       prominence=np.std(smoothed_saliency) * 0.5)
    
    if len(peaks) > 0:
        peak_scatter = ax.scatter(peaks, smoothed_saliency[peaks], 
                                 color=jc.custom_colors['raspberry'], s=40, zorder=5, 
                                 edgecolors='white', linewidth=1,
                                 label=f'Peaks ({len(peaks)} found)')
        
        for i, peak in enumerate(peaks[:5]):  # Limit to first 5 peaks
            ax.annotate(f'{peak}', 
                       (peak, smoothed_saliency[peak]), 
                       xytext=(0, 8), textcoords='offset points', 
                       ha='center', va='bottom',
                       fontsize=8, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', 
                                facecolor='white', 
                                edgecolor=jc.custom_colors['raspberry'],
                                linewidth=0.5,
                                alpha=0.9))
    
    if show_context and ids is not None and tokenizer is not None and len(peaks) > 0:
        _add_context_annotations(ax, peaks, smoothed_saliency, ids, tokenizer, 
                               context_window)
    
    ax.set_title('Smoothed Saliency Analysis', loc='left')
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Saliency Score')
    ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray', 
             borderpad=0.3, handletextpad=0.5)
    ax.margins(x=0.01, y=0.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='svg')
    plt.close()
    
    return fig, peaks, smoothed_saliency


def _add_context_annotations(ax, peaks, smoothed_saliency, ids, tokenizer, context_window):
    """Helper function to add token context annotations for peaks"""
    

    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    
    max_annotations = 5
    peaks_to_annotate = peaks[:max_annotations]
    
    peak_importance = [(p, smoothed_saliency[p]) for p in peaks]
    peak_importance.sort(key=lambda x: x[1], reverse=True)
    peaks_to_annotate = [p[0] for p in peak_importance[:max_annotations]]
    peaks_to_annotate.sort()  # Sort by position for left-to-right reading
    
    annotation_positions = []
    
    for i, peak in enumerate(peaks_to_annotate):
        start = max(0, peak - context_window)
        end = min(len(ids), peak + context_window + 1)
        
        context_ids = ids[start:end]
        context_positions = list(range(start, end))
        
        context_tokens = []
        for token_id in context_ids:
            if token_id == 0:  # Skip padding
                token = "[PAD]"
            elif hasattr(tokenizer, 'convert_ids_to_tokens'):
                token = tokenizer.convert_ids_to_tokens(int(token_id))
                if isinstance(token, str) and token.startswith('##'):
                    token = token[2:]
            elif hasattr(tokenizer, 'decode'):
                token = tokenizer.decode([int(token_id)]).strip()
            else:
                token = str(token_id)
            context_tokens.append(token)
        peak_idx = context_positions.index(peak) if peak in context_positions else -1
        
        if peak_idx >= 0 and peak_idx < len(context_tokens):
            display_window = 2  
            start_idx = max(0, peak_idx - display_window)
            end_idx = min(len(context_tokens), peak_idx + display_window + 1)
            
            display_tokens = context_tokens[start_idx:end_idx]
            display_peak_idx = peak_idx - start_idx
            
            before = ' '.join(display_tokens[:display_peak_idx])
            peak_token = display_tokens[display_peak_idx]
            after = ' '.join(display_tokens[display_peak_idx + 1:])
            
            if start_idx > 0:
                before = "... " + before
            if end_idx < len(context_tokens):
                after = after + " ..."
                
            context_str = f"{before} [{peak_token}] {after}".strip()
        else:
            context_str = ' '.join(context_tokens[:5]) + "..."
        
        if len(context_str) > 30:
            context_str = context_str[:27] + "..."
        
        base_offset = 0.08 * y_range

        if i % 2 == 0:
            y_offset = base_offset
            connection_style = 'arc3,rad=0.2'
        else:
            y_offset = -base_offset * 0.7
            connection_style = 'arc3,rad=-0.2'
        
        y_pos = smoothed_saliency[peak] + y_offset
        for prev_peak, prev_y in annotation_positions:
            if abs(peak - prev_peak) < 50 and abs(y_pos - prev_y) < 0.04 * y_range:
                y_pos += 0.05 * y_range * (1 if y_offset > 0 else -1)
        
        annotation_positions.append((peak, y_pos))
        
        ax.annotate(context_str, 
                   (peak, smoothed_saliency[peak]),
                   xytext=(peak, y_pos),
                   arrowprops=dict(arrowstyle='->', 
                                 connectionstyle=connection_style,
                                 color=jc.custom_colors['julich_blue_1'],
                                 linewidth=0.8,
                                 alpha=0.7),
                   bbox=dict(boxstyle='round,pad=0.2', 
                            facecolor='white', 
                            edgecolor=jc.custom_colors['julich_blue_1'],
                            linewidth=0.5,
                            alpha=0.95),
                   fontsize=8,  # Reduced from 30
                   ha='center',
                   fontfamily='monospace')
    
    if len(annotation_positions) > 0:
        current_ylim = ax.get_ylim()
        y_range = current_ylim[1] - current_ylim[0]
        ax.set_ylim(current_ylim[0] - 0.1 * y_range, current_ylim[1] + 0.15 * y_range)