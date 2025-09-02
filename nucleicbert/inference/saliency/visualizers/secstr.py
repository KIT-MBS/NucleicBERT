import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import juelich_colors

def set_style():
    font = {
            'family' : 'sans-serif',
            'size'   : 25
    }
    axes = {
            'titlesize' : 25,
            'labelsize' : 25,
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

def plot_saliency_structure_correlation(saliency_map, sequence, structure_matrix, save_path, 
                                      figsize=(7, 3), palette_type='primary'):
    """
    Create saliency plot with paired regions shown as bars, unpaired regions have no bars
    
    Args:
        saliency_map (torch.Tensor): 1D saliency map tensor
        sequence (str): RNA sequence (AUGC)
        structure_matrix (torch.Tensor or np.ndarray): Binary matrix where 1=paired, 0=unpaired [seq_len, seq_len]
        save_path (str): Path to save the visualization
        figsize (tuple): Figure size (width, height)
        palette_type (str): Color palette type ('primary', 'secondary', 'all')
    """
    
    juelich_colors.set_matplotlib_color_cycle(palette_type)
    
    if hasattr(saliency_map, 'detach'):
        saliency = saliency_map.squeeze().detach().cpu().numpy()
    else:
        saliency = np.array(saliency_map)
    
    if hasattr(structure_matrix, 'detach'):
        struct_matrix = structure_matrix.squeeze().detach().cpu().numpy()
    else:
        struct_matrix = np.array(structure_matrix)
    

    if np.max(saliency) == 0:
        saliency = np.ones_like(saliency) * 0.01
    

    sequence = str(sequence)
    

    min_len = min(len(sequence), saliency.shape[-1], struct_matrix.shape[0])
    sequence = sequence[:min_len]
    saliency = saliency[:min_len]
    struct_matrix = struct_matrix[:min_len, :min_len]
    

    paired_positions = set()
    upper_triangle = np.triu(struct_matrix, k=1)  # k=1 to exclude diagonal
    

    pairs = np.where(upper_triangle == 1)
    for i, j in zip(pairs[0], pairs[1]):
        paired_positions.add(i)
        paired_positions.add(j)
    
    paired_positions = sorted(list(paired_positions))
    

    if np.max(saliency) > 0:
        norm_saliency = saliency / np.max(saliency)
    else:
        norm_saliency = saliency
    

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    

    positions = np.arange(len(saliency))
    

    ax.plot(positions, norm_saliency, 
            color=juelich_colors.custom_colors['julich_blue_1'], 
            linewidth=2.5, 
            label='Saliency', 
            alpha=0.8)
    ax.fill_between(positions, norm_saliency, alpha=0.2, 
                    color=juelich_colors.custom_colors['julich_blue_2'])

    if paired_positions:

        paired_mask = np.zeros_like(norm_saliency)
        paired_mask[paired_positions] = 1.0
        

        ax.fill_between(positions, 0, paired_mask, 
                       color=juelich_colors.custom_colors['raspberry'],
                       edgecolor=juelich_colors.custom_colors['raspberry'],
                       alpha=0.2,
                       step='mid',
                       linewidth=0,
                       label='Paired regions')
    

    ax.set_title('Saliency with Paired Regions', loc='left')
    ax.set_xlabel('Position')
    ax.set_ylabel('Normalized Saliency')
    ax.set_xlim(0, len(saliency))
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.margins(x=0.01, y=0.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, transparent=True, format='svg')
    plt.close()
    
    return fig