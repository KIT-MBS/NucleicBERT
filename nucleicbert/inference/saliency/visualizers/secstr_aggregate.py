import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from typing import List, Dict, Optional, Tuple, Union
from itertools import combinations
import scipy.stats as stats

def analyze_saliency_vs_secstr_properties(
    saliency_maps: List[torch.Tensor],
    sequences: List[str],
    structures: List[str],
    save_dir: str,
    prefix: str = "secstr_aggregate"
) -> Dict:
    """
    Analyze the relationship between saliency and various secondary structure properties
    across multiple RNA samples
    
    Args:
        saliency_maps: List of saliency map tensors
        sequences: List of RNA sequences
        structures: List of secondary structures in dot-bracket notation
        save_dir: Directory to save the visualizations
        prefix: Prefix for saved files
        
    Returns:
        Dictionary of analysis results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    sal_maps_np = []
    for sal in saliency_maps:
        if hasattr(sal, 'detach'):
            sal_maps_np.append(sal.squeeze().detach().cpu().numpy())
        else:
            sal_maps_np.append(np.array(sal).squeeze())
    
    property_data = []
    
    for i, (sal, sequence, structure) in enumerate(zip(sal_maps_np, sequences, structures)):
        min_len = min(len(sal), len(sequence), len(structure))
        sal = sal[:min_len]
        sequence = sequence[:min_len]
        structure = structure[:min_len]
        
        stack = []
        paired_positions = set()
        loop_positions = set()
        stem_positions = set()
        hairpin_positions = set()
        bulge_positions = set()
        internal_loop_positions = set()
        paired_to = {}  # Maps position to its paired position
        
        for j, char in enumerate(structure):
            if char == '(':
                stack.append(j)
            elif char == ')':
                if stack:
                    left_pos = stack.pop()
                    paired_positions.add(left_pos)
                    paired_positions.add(j)
                    paired_to[left_pos] = j
                    paired_to[j] = left_pos
        
        in_hairpin = False
        hairpin_start = -1
        
        for j in range(len(structure)):
            if j in paired_positions:
                # Check if it's part of a stem
                if j > 0 and j-1 in paired_positions and paired_to.get(j-1, -1) == paired_to.get(j, -1) + 1:
                    stem_positions.add(j)
                # Start of a hairpin
                if structure[j] == '(' and j+1 < len(structure) and structure[j+1] == '.':
                    in_hairpin = True
                    hairpin_start = j
                # End of a hairpin
                if structure[j] == ')' and in_hairpin and paired_to.get(j) == hairpin_start:
                    in_hairpin = False
                    hairpin_start = -1
            else:  # Unpaired position
                # Check if it's in a loop
                if j > 0 and j < len(structure) - 1:
                    if j-1 in paired_positions and j+1 in paired_positions:
                        # Internal loop
                        if paired_to.get(j-1) > paired_to.get(j+1):
                            internal_loop_positions.add(j)
                        # Bulge
                        else:
                            bulge_positions.add(j)
                    elif in_hairpin:
                        hairpin_positions.add(j)
                    else:
                        loop_positions.add(j)
                else:
                    # Terminal positions are considered loops
                    loop_positions.add(j)
        
        for j in range(min_len):
            is_paired = j in paired_positions
            structure_type = "unknown"
            if j in stem_positions:
                structure_type = "stem"
            elif j in hairpin_positions:
                structure_type = "hairpin"
            elif j in bulge_positions:
                structure_type = "bulge"
            elif j in internal_loop_positions:
                structure_type = "internal_loop"
            elif j in loop_positions:
                structure_type = "external_loop"
            elif is_paired:
                structure_type = "paired"  # Other paired positions not in stems
            
            min_dist_to_hairpin = min([abs(j-h) for h in hairpin_positions], default=min_len)
            
            property_data.append({
                'sample': i,
                'position': j,
                'nucleotide': sequence[j],
                'saliency': sal[j],
                'is_paired': is_paired,
                'structure_type': structure_type,
                'distance_to_hairpin': min_dist_to_hairpin,
                'is_g': sequence[j] == 'G',
                'is_c': sequence[j] == 'C',
                'is_a': sequence[j] == 'A',
                'is_u': sequence[j] == 'U'
            })
    
    df = pd.DataFrame(property_data)
    
    df.to_csv(os.path.join(save_dir, f"{prefix}_saliency_secstr_data.csv"), index=False)
    
    # 1. Box plot comparing saliency for paired vs unpaired positions
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='is_paired', y='saliency', data=df)
    plt.xlabel('Is Paired')
    plt.ylabel('Saliency')
    plt.title('Saliency Distribution: Paired vs. Unpaired Nucleotides')
    plt.xticks([0, 1], ['No', 'Yes'])
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_saliency_by_paired.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Box plot comparing saliency across structure types
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='structure_type', y='saliency', data=df, order=['stem', 'hairpin', 'bulge', 'internal_loop', 'external_loop', 'paired'])
    plt.xlabel('Structure Type')
    plt.ylabel('Saliency')
    plt.title('Saliency Distribution by Secondary Structure Element')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_saliency_by_structure_type.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Scatter plot of saliency vs. distance to nearest hairpin
    plt.figure(figsize=(10, 6))
    sns.regplot(x='distance_to_hairpin', y='saliency', data=df, scatter_kws={'alpha': 0.3})
    plt.xlabel('Distance to Nearest Hairpin')
    plt.ylabel('Saliency')
    plt.title('Saliency vs. Distance to Nearest Hairpin')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_saliency_vs_hairpin_distance.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Box plot comparing saliency across nucleotide types
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='nucleotide', y='saliency', data=df, order=['A', 'U', 'G', 'C'])
    plt.xlabel('Nucleotide')
    plt.ylabel('Saliency')
    plt.title('Saliency Distribution by Nucleotide Type')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_saliency_by_nucleotide.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Correlation heatmap
    structure_dummies = pd.get_dummies(df['structure_type'], prefix='structure')
    df_corr = pd.concat([df[['saliency', 'is_paired', 'distance_to_hairpin', 'is_g', 'is_c', 'is_a', 'is_u']], structure_dummies], axis=1)
    corr_df = df_corr.corr()
    
    saliency_corr = corr_df['saliency'].sort_values(ascending=False)
    
    plt.figure(figsize=(10, 8))
    plt.bar(x=saliency_corr.index[1:], height=saliency_corr.values[1:])  # Skip first (self-correlation)
    plt.xticks(rotation=90)
    plt.xlabel('Feature')
    plt.ylabel('Correlation with Saliency')
    plt.title('Correlation of Features with Saliency')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_correlation_bar_chart.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Group bar chart showing average saliency by structure type
    means = df.groupby('structure_type')['saliency'].mean().reindex(['stem', 'hairpin', 'bulge', 'internal_loop', 'external_loop', 'paired'])
    stds = df.groupby('structure_type')['saliency'].std().reindex(['stem', 'hairpin', 'bulge', 'internal_loop', 'external_loop', 'paired'])
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(means.index, means.values, yerr=stds.values, capsize=10)
    plt.xlabel('Secondary Structure Element')
    plt.ylabel('Average Saliency')
    plt.title('Average Saliency by Secondary Structure Element')
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01, 
                f'{means.values[i]:.3f}', 
                ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_avg_saliency_by_structure.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    try:
        # T-test between paired vs. unpaired
        paired_saliency = df[df['is_paired']]['saliency']
        unpaired_saliency = df[~df['is_paired']]['saliency']
        
        t_stat_paired, p_val_paired = stats.ttest_ind(
            paired_saliency, 
            unpaired_saliency, 
            equal_var=False
        )
        
        # One-way ANOVA for structure types
        structure_groups = [df[df['structure_type'] == st]['saliency'] for st in 
                          ['stem', 'hairpin', 'bulge', 'internal_loop', 'external_loop', 'paired']
                          if sum(df['structure_type'] == st) > 0]
        
        # Only perform ANOVA if we have at least 2 groups
        if len(structure_groups) >= 2:
            f_stat, p_val_anova = stats.f_oneway(*structure_groups)
        else:
            f_stat, p_val_anova = np.nan, np.nan
        
        # Correlations
        corr_hairpin_dist, p_hairpin = stats.pearsonr(df['distance_to_hairpin'], df['saliency'])
        
        # Save statistical results
        with open(os.path.join(save_dir, f"{prefix}_statistical_tests.txt"), 'w') as f:
            f.write("Statistical Tests for Saliency vs. Secondary Structure Properties\n")
            f.write("==============================================================\n\n")
            
            f.write("T-tests:\n")
            f.write(f"Paired vs. Unpaired: t={t_stat_paired:.4f}, p={p_val_paired:.4e}\n\n")
            
            f.write("ANOVA:\n")
            if not np.isnan(f_stat):
                f.write(f"Structure Types: F={f_stat:.4f}, p={p_val_anova:.4e}\n\n")
            else:
                f.write("Structure Types: Insufficient data for ANOVA\n\n")
            
            f.write("Correlations:\n")
            f.write(f"Saliency vs. Distance to Hairpin: r={corr_hairpin_dist:.4f}, p={p_hairpin:.4e}\n")
            
            f.write("\nSummary Statistics:\n")
            f.write(f"Mean Saliency (Paired): {paired_saliency.mean():.6f}\n")
            f.write(f"Mean Saliency (Unpaired): {unpaired_saliency.mean():.6f}\n\n")
            
            f.write("Mean Saliency by Structure Type:\n")
            for st in ['stem', 'hairpin', 'bulge', 'internal_loop', 'external_loop', 'paired']:
                if sum(df['structure_type'] == st) > 0:
                    f.write(f"{st}: {df[df['structure_type'] == st]['saliency'].mean():.6f}\n")
            
            f.write("\nMean Saliency by Nucleotide:\n")
            for nt in ['A', 'U', 'G', 'C']:
                if sum(df['nucleotide'] == nt) > 0:
                    f.write(f"{nt}: {df[df['nucleotide'] == nt]['saliency'].mean():.6f}\n")
        
        stats_results = {
            't_paired': t_stat_paired,
            'p_paired': p_val_paired,
            'f_structure': f_stat,
            'p_structure': p_val_anova,
            'corr_hairpin_dist': corr_hairpin_dist,
            'p_hairpin_dist': p_hairpin
        }
    except Exception as e:
        print(f"Error in statistical tests: {e}")
        stats_results = {}
    
    return {
        'data': df,
        'stats': stats_results,
        'means': means.to_dict(),
        'stds': stds.to_dict()
    }

def plot_normalized_saliency_threshold_secstr(
    saliency_maps: List[torch.Tensor],
    sequences: List[str],
    structures: List[str],
    save_dir: str,
    prefix: str = "secstr_aggregate"
) -> None:
    """
    Analyze the relationship between normalized saliency thresholds and secondary structure properties
    
    Args:
        saliency_maps: List of saliency map tensors
        sequences: List of RNA sequences
        structures: List of secondary structures in dot-bracket notation
        save_dir: Directory to save the visualizations
        prefix: Prefix for saved files
    """
    os.makedirs(save_dir, exist_ok=True)
    
    norm_sal_maps = []
    for sal in saliency_maps:
        if hasattr(sal, 'detach'):
            sal_np = sal.squeeze().detach().cpu().numpy()
        else:
            sal_np = np.array(sal).squeeze()
        
        max_val = np.max(sal_np)
        if max_val > 0:
            norm_sal_maps.append(sal_np / max_val)
        else:
            norm_sal_maps.append(sal_np)
    
    is_paired = []
    structure_types = []
    nucleotides = []
    norm_saliency = []
    
    for sal, sequence, structure in zip(norm_sal_maps, sequences, structures):
        min_len = min(len(sal), len(sequence), len(structure))
        stack = []
        paired_positions = set()
        paired_to = {}
        
        for i in range(min_len):
            if structure[i] == '(':
                stack.append(i)
            elif structure[i] == ')':
                if stack:
                    left_pos = stack.pop()
                    paired_positions.add(left_pos)
                    paired_positions.add(i)
                    paired_to[left_pos] = i
                    paired_to[i] = left_pos
        
        for i in range(min_len):
            is_paired.append(i in paired_positions)
            nucleotides.append(sequence[i])
            norm_saliency.append(sal[i])
            
            # Determine structure type
            if i in paired_positions:
                # Check if it's part of a stem (both adjacent positions are paired)
                if (i > 0 and i-1 in paired_positions and 
                    paired_to.get(i-1, -1) == paired_to.get(i, -1) + 1):
                    structure_types.append('stem')
                else:
                    structure_types.append('paired')
            else:
                # Unpaired - check if it's in a loop between paired regions
                if i > 0 and i < min_len-1:
                    if i-1 in paired_positions and i+1 in paired_positions:
                        if paired_to.get(i-1) > paired_to.get(i+1):
                            structure_types.append('internal_loop')
                        else:
                            structure_types.append('bulge')
                    # Check if it's in a hairpin loop
                    elif i-1 in paired_positions and structure[i-1] == '(':
                        # Look ahead to find closing parenthesis
                        closing_pos = paired_to.get(i-1, -1)
                        if closing_pos > i:
                            structure_types.append('hairpin')
                        else:
                            structure_types.append('external_loop')
                    else:
                        structure_types.append('external_loop')
                else:
                    structure_types.append('external_loop')
    
    thresholds = np.linspace(0, 1, 101)  
    
    paired_fractions = []
    unpaired_fractions = []
    stem_fractions = []
    hairpin_fractions = []
    loop_fractions = []  # Combined internal, external loops and bulges
    a_fractions = []
    u_fractions = []
    g_fractions = []
    c_fractions = []
    
    for threshold in thresholds:
        high_saliency = np.array(norm_saliency) >= threshold
        
        paired_mask = np.array(is_paired)
        unpaired_mask = ~paired_mask
        stem_mask = np.array([st == 'stem' for st in structure_types])
        hairpin_mask = np.array([st == 'hairpin' for st in structure_types])
        loop_mask = np.array([st in ['internal_loop', 'external_loop', 'bulge'] for st in structure_types])
        
        a_mask = np.array([nt == 'A' for nt in nucleotides])
        u_mask = np.array([nt == 'U' for nt in nucleotides])
        g_mask = np.array([nt == 'G' for nt in nucleotides])
        c_mask = np.array([nt == 'C' for nt in nucleotides])
        
        paired_fractions.append(np.mean(high_saliency[paired_mask]) if sum(paired_mask) > 0 else 0)
        unpaired_fractions.append(np.mean(high_saliency[unpaired_mask]) if sum(unpaired_mask) > 0 else 0)
        stem_fractions.append(np.mean(high_saliency[stem_mask]) if sum(stem_mask) > 0 else 0)
        hairpin_fractions.append(np.mean(high_saliency[hairpin_mask]) if sum(hairpin_mask) > 0 else 0)
        loop_fractions.append(np.mean(high_saliency[loop_mask]) if sum(loop_mask) > 0 else 0)
        
        a_fractions.append(np.mean(high_saliency[a_mask]) if sum(a_mask) > 0 else 0)
        u_fractions.append(np.mean(high_saliency[u_mask]) if sum(u_mask) > 0 else 0)
        g_fractions.append(np.mean(high_saliency[g_mask]) if sum(g_mask) > 0 else 0)
        c_fractions.append(np.mean(high_saliency[c_mask]) if sum(c_mask) > 0 else 0)
    

    plt.figure(figsize=(12, 8))
    plt.plot(thresholds, paired_fractions, 'b-', label='Paired')
    plt.plot(thresholds, unpaired_fractions, 'r-', label='Unpaired')
    plt.plot(thresholds, stem_fractions, 'g-', label='Stem')
    plt.plot(thresholds, hairpin_fractions, 'm-', label='Hairpin')
    plt.plot(thresholds, loop_fractions, 'c-', label='Loops (Internal/External/Bulge)')
    
    plt.xlabel('Normalized Saliency Threshold')
    plt.ylabel('Fraction of Positions')
    plt.title('Fraction of Positions Above Saliency Threshold by Structure Type')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_saliency_threshold_structure.png"), dpi=300, bbox_inches='tight')
    plt.close()
    

    plt.figure(figsize=(12, 8))
    plt.plot(thresholds, a_fractions, 'g-', label='A')
    plt.plot(thresholds, u_fractions, 'r-', label='U')
    plt.plot(thresholds, g_fractions, 'b-', label='G')
    plt.plot(thresholds, c_fractions, 'y-', label='C')
    
    plt.xlabel('Normalized Saliency Threshold')
    plt.ylabel('Fraction of Positions')
    plt.title('Fraction of Positions Above Saliency Threshold by Nucleotide')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_saliency_threshold_nucleotide.png"), dpi=300, bbox_inches='tight')
    plt.close()
    

    threshold_data = pd.DataFrame({
        'threshold': thresholds,
        'paired': paired_fractions,
        'unpaired': unpaired_fractions,
        'stem': stem_fractions,
        'hairpin': hairpin_fractions,
        'loops': loop_fractions,
        'A': a_fractions,
        'U': u_fractions,
        'G': g_fractions,
        'C': c_fractions
    })
    
    threshold_data.to_csv(os.path.join(save_dir, f"{prefix}_saliency_threshold_data.csv"), index=False)

def run_all_secstr_analyses(
    saliency_maps: List[torch.Tensor],
    sequences: List[str],
    structures: List[str],
    save_dir: str,
    prefix: str = "secstr_aggregate"
) -> Dict:
    """
    Run all secondary structure aggregate analyses on a set of saliency maps
    
    Args:
        saliency_maps: List of saliency map tensors
        sequences: List of RNA sequences
        structures: List of secondary structures in dot-bracket notation
        save_dir: Directory to save all visualizations
        prefix: Prefix for saved files
        
    Returns:
        Dictionary with all analysis results
    """
    os.makedirs(save_dir, exist_ok=True)
    property_analysis = analyze_saliency_vs_secstr_properties(
        saliency_maps, sequences, structures, save_dir, prefix
    )
    
    plot_normalized_saliency_threshold_secstr(
        saliency_maps, sequences, structures, save_dir, prefix
    )
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>RNA Secondary Structure Saliency Analysis</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333; }}
            .plot-container {{ margin: 20px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}
            img {{ max-width: 100%; height: auto; }}
            .summary {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>RNA Secondary Structure Saliency Analysis Results</h1>
        
        <div class="summary">
            <h2>Summary Statistics</h2>
            <p>
                Number of Samples: {len(saliency_maps)}<br>
                Number of Sequences: {len(sequences)}<br>
                Number of Structures: {len(structures)}<br>
            </p>
        </div>
        
        <h2>Secondary Structure Property Analysis</h2>
        
        <div class="plot-container">
            <h3>Saliency Distribution: Paired vs. Unpaired</h3>
            <img src="{prefix}_saliency_by_paired.png" alt="Saliency by Paired Status">
            <p>Box plot comparing saliency distributions for paired and unpaired positions.</p>
        </div>
        
        <div class="plot-container">
            <h3>Saliency Distribution by Secondary Structure Element</h3>
            <img src="{prefix}_saliency_by_structure_type.png" alt="Saliency by Structure Type">
            <p>Box plot comparing saliency distributions across different secondary structure elements.</p>
        </div>
        
        <div class="plot-container">
            <h3>Saliency vs. Distance to Nearest Hairpin</h3>
            <img src="{prefix}_saliency_vs_hairpin_distance.png" alt="Saliency vs Hairpin Distance">
            <p>Scatter plot with regression line showing relationship between distance to nearest hairpin and saliency.</p>
        </div>
        
        <div class="plot-container">
            <h3>Saliency Distribution by Nucleotide Type</h3>
            <img src="{prefix}_saliency_by_nucleotide.png" alt="Saliency by Nucleotide">
            <p>Box plot comparing saliency distributions across different nucleotide types.</p>
        </div>
        
        <div class="plot-container">
            <h3>Correlation of Features with Saliency</h3>
            <img src="{prefix}_correlation_bar_chart.png" alt="Correlation Bar Chart">
            <p>Bar chart showing correlations between various features and saliency values.</p>
        </div>
        
        <div class="plot-container">
            <h3>Average Saliency by Secondary Structure Element</h3>
            <img src="{prefix}_avg_saliency_by_structure.png" alt="Average Saliency by Structure">
            <p>Bar chart showing the average saliency for different types of secondary structure elements.</p>
        </div>
        
        <h2>Threshold Analysis</h2>
        
        <div class="plot-container">
            <h3>Fraction of Positions Above Saliency Threshold by Structure Type</h3>
            <img src="{prefix}_saliency_threshold_structure.png" alt="Saliency Threshold by Structure">
            <p>Line plot showing how different structure types relate to saliency thresholds.</p>
        </div>
        
        <div class="plot-container">
            <h3>Fraction of Positions Above Saliency Threshold by Nucleotide</h3>
            <img src="{prefix}_saliency_threshold_nucleotide.png" alt="Saliency Threshold by Nucleotide">
            <p>Line plot showing how different nucleotides relate to saliency thresholds.</p>
        </div>
        
        <footer>
            <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </footer>
    </body>
    </html>
    """
    
    with open(os.path.join(save_dir, f"{prefix}_analysis_index.html"), 'w') as f:
        f.write(html_content)
    
    return {
        'property_analysis': property_analysis
    }

def compare_saliency_across_rna_properties(
    saliency_maps: List[torch.Tensor],
    sequences: List[str],
    structures: List[str],
    save_dir: str,
    prefix: str = "secstr_aggregate"
) -> Dict:
    """
    Compare saliency maps across different RNA properties like GC content, 
    structure complexity, and sequence motifs
    
    Args:
        saliency_maps: List of saliency map tensors
        sequences: List of RNA sequences
        structures: List of secondary structures in dot-bracket notation
        save_dir: Directory to save visualizations
        prefix: Prefix for saved files
        
    Returns:
        Dictionary with analysis results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    rna_properties = []
    
    for i, (sal, sequence, structure) in enumerate(zip(saliency_maps, sequences, structures)):
        if hasattr(sal, 'detach'):
            sal_np = sal.squeeze().detach().cpu().numpy()
        else:
            sal_np = np.array(sal).squeeze()
        
        min_len = min(len(sal_np), len(sequence), len(structure))
        sal_np = sal_np[:min_len]
        sequence = sequence[:min_len]
        structure = structure[:min_len]
        
        gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
        paired_ratio = structure.count('(') * 2 / len(structure)  # x2 because each '(' has a ')'
        
        stem_count = 0
        hairpin_count = 0
        bulge_count = 0
        internal_loop_count = 0

        stack = []
        paired_positions = set()
        paired_to = {}
        
        # First identify paired positions
        for j, char in enumerate(structure):
            if char == '(':
                stack.append(j)
            elif char == ')':
                if stack:
                    left_pos = stack.pop()
                    paired_positions.add(left_pos)
                    paired_positions.add(j)
                    paired_to[left_pos] = j
                    paired_to[j] = left_pos
        
        # Count stems (continuous paired regions)
        in_stem = False
        for j in range(1, len(structure)):
            if j in paired_positions and j-1 in paired_positions:
                if not in_stem:
                    stem_count += 1
                    in_stem = True
            else:
                in_stem = False
        
        # Count hairpins, bulges, and internal loops
        j = 0
        while j < len(structure):
            if structure[j] == '(':
                # Look for hairpins (unpaired region between matched brackets)
                closing_pos = paired_to.get(j, -1)
                if closing_pos > j + 1:
                    if all(c == '.' for c in structure[j+1:closing_pos]):
                        hairpin_count += 1
                    
                    # Look for bulges and internal loops
                    k = j + 1
                    while k < closing_pos:
                        if structure[k] == '.':
                            # Count consecutive unpaired nucleotides
                            unpaired_start = k
                            while k < closing_pos and structure[k] == '.':
                                k += 1
                            unpaired_len = k - unpaired_start
                            
                            # Check if it's part of a bulge or internal loop
                            if k < closing_pos and structure[k] == '(':
                                if unpaired_len == 1:
                                    bulge_count += 1
                                else:
                                    internal_loop_count += 1
                        elif structure[k] == '(':
                            # Skip to the matching closing bracket
                            k = paired_to.get(k, k+1)
                        k += 1
            j += 1
        

        complexity_score = stem_count + hairpin_count + bulge_count + internal_loop_count
        

        mean_saliency = np.mean(sal_np)
        max_saliency = np.max(sal_np)
        
        gu_wobble_count = 0
        gnra_tetraloop_count = 0
        
        for j in paired_positions:
            if j in paired_to:
                paired_j = paired_to[j]
                if j < paired_j:  # Only count each pair once
                    if j < len(sequence) and paired_j < len(sequence):
                        if (sequence[j] == 'G' and sequence[paired_j] == 'U') or \
                           (sequence[j] == 'U' and sequence[paired_j] == 'G'):
                            gu_wobble_count += 1
        
        for j in range(len(sequence) - 3):
            if j+3 < len(sequence):
                motif = sequence[j:j+4]
                if motif[0] == 'G' and motif[1] in 'AUGC' and motif[2] == 'A' and motif[3] in 'AG':
                    # Check if it's in a loop
                    if all(c == '.' for c in structure[j:j+4]):
                        gnra_tetraloop_count += 1
        
        rna_properties.append({
            'sample': i,
            'length': min_len,
            'gc_content': gc_content,
            'paired_ratio': paired_ratio,
            'stem_count': stem_count,
            'hairpin_count': hairpin_count,
            'bulge_count': bulge_count,
            'internal_loop_count': internal_loop_count,
            'complexity_score': complexity_score,
            'gu_wobble_count': gu_wobble_count,
            'gnra_tetraloop_count': gnra_tetraloop_count,
            'mean_saliency': mean_saliency,
            'max_saliency': max_saliency
        })
    
    rna_df = pd.DataFrame(rna_properties)
    
    rna_df.to_csv(os.path.join(save_dir, f"{prefix}_rna_properties.csv"), index=False)
    
    
    # 1. Scatter plot matrix of RNA properties vs saliency
    plt.figure(figsize=(15, 15))
    
    properties = ['gc_content', 'paired_ratio', 'complexity_score', 'stem_count', 
                  'hairpin_count', 'mean_saliency', 'max_saliency']
    
    pd.plotting.scatter_matrix(rna_df[properties], alpha=0.8, diagonal='kde', figsize=(15, 15))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_property_scatter_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Correlation heatmap between RNA properties
    plt.figure(figsize=(12, 10))
    corr = rna_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', vmin=-1, vmax=1, 
                annot=True, fmt='.2f', square=True, linewidths=.5)
    plt.title('Correlation Between RNA Properties and Saliency')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_property_correlation_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Scatter plots for key relationships
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # GC content vs mean saliency
    sns.regplot(x='gc_content', y='mean_saliency', data=rna_df, ax=axs[0, 0])
    axs[0, 0].set_title('GC Content vs Mean Saliency')
    axs[0, 0].set_xlabel('GC Content')
    axs[0, 0].set_ylabel('Mean Saliency')
    
    # Paired ratio vs mean saliency
    sns.regplot(x='paired_ratio', y='mean_saliency', data=rna_df, ax=axs[0, 1])
    axs[0, 1].set_title('Paired Ratio vs Mean Saliency')
    axs[0, 1].set_xlabel('Paired Ratio')
    axs[0, 1].set_ylabel('Mean Saliency')
    
    # Complexity score vs mean saliency
    sns.regplot(x='complexity_score', y='mean_saliency', data=rna_df, ax=axs[1, 0])
    axs[1, 0].set_title('Structure Complexity vs Mean Saliency')
    axs[1, 0].set_xlabel('Complexity Score')
    axs[1, 0].set_ylabel('Mean Saliency')
    
    # GU wobble count vs mean saliency
    sns.regplot(x='gu_wobble_count', y='mean_saliency', data=rna_df, ax=axs[1, 1])
    axs[1, 1].set_title('GU Wobble Pairs vs Mean Saliency')
    axs[1, 1].set_xlabel('GU Wobble Pair Count')
    axs[1, 1].set_ylabel('Mean Saliency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_key_property_relationships.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Bar chart comparing saliency for sequences with and without specific motifs
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    
    # GU wobble presence
    has_gu = rna_df['gu_wobble_count'] > 0
    gu_groups = [rna_df[~has_gu]['mean_saliency'], rna_df[has_gu]['mean_saliency']]
    axs[0].bar(['No GU Wobbles', 'Has GU Wobbles'], 
              [np.mean(gu_groups[0]) if len(gu_groups[0]) > 0 else 0, 
               np.mean(gu_groups[1]) if len(gu_groups[1]) > 0 else 0])
    axs[0].set_title('Mean Saliency by GU Wobble Presence')
    axs[0].set_ylabel('Mean Saliency')
    
    # GNRA tetraloop presence
    has_gnra = rna_df['gnra_tetraloop_count'] > 0
    gnra_groups = [rna_df[~has_gnra]['mean_saliency'], rna_df[has_gnra]['mean_saliency']]
    axs[1].bar(['No GNRA Tetraloops', 'Has GNRA Tetraloops'], 
              [np.mean(gnra_groups[0]) if len(gnra_groups[0]) > 0 else 0, 
               np.mean(gnra_groups[1]) if len(gnra_groups[1]) > 0 else 0])
    axs[1].set_title('Mean Saliency by GNRA Tetraloop Presence')
    axs[1].set_ylabel('Mean Saliency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_motif_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    try:
        # T-test for GU wobble presence
        if sum(has_gu) > 0 and sum(~has_gu) > 0:
            t_gu, p_gu = stats.ttest_ind(
                rna_df[has_gu]['mean_saliency'],
                rna_df[~has_gu]['mean_saliency'],
                equal_var=False
            )
        else:
            t_gu, p_gu = np.nan, np.nan
            
        # T-test for GNRA tetraloop presence
        if sum(has_gnra) > 0 and sum(~has_gnra) > 0:
            t_gnra, p_gnra = stats.ttest_ind(
                rna_df[has_gnra]['mean_saliency'],
                rna_df[~has_gnra]['mean_saliency'],
                equal_var=False
            )
        else:
            t_gnra, p_gnra = np.nan, np.nan
            
        with open(os.path.join(save_dir, f"{prefix}_rna_property_tests.txt"), 'w') as f:
            f.write("Statistical Tests for RNA Properties vs. Saliency\n")
            f.write("=============================================\n\n")
            
            f.write("Sample Statistics:\n")
            f.write(f"Number of RNA sequences: {len(rna_df)}\n")
            f.write(f"Average sequence length: {rna_df['length'].mean():.2f}\n")
            f.write(f"Average GC content: {rna_df['gc_content'].mean():.2f}\n")
            f.write(f"Average paired ratio: {rna_df['paired_ratio'].mean():.2f}\n")
            f.write(f"Average structure complexity: {rna_df['complexity_score'].mean():.2f}\n\n")
            
            f.write("Correlations with Mean Saliency:\n")
            for prop in ['gc_content', 'paired_ratio', 'complexity_score', 'stem_count', 
                        'hairpin_count', 'bulge_count', 'internal_loop_count']:
                corr_val, p_val = stats.pearsonr(rna_df[prop], rna_df['mean_saliency'])
                f.write(f"{prop}: r={corr_val:.4f}, p={p_val:.4e}\n")
            
            f.write("\nMotif Analysis:\n")
            if not np.isnan(t_gu):
                f.write(f"GU Wobbles: t={t_gu:.4f}, p={p_gu:.4e}\n")
                f.write(f"  Mean saliency with GU wobbles: {np.mean(gu_groups[1]):.4f}\n")
                f.write(f"  Mean saliency without GU wobbles: {np.mean(gu_groups[0]):.4f}\n")
            else:
                f.write("GU Wobbles: Insufficient data for t-test\n")
                
            if not np.isnan(t_gnra):
                f.write(f"GNRA Tetraloops: t={t_gnra:.4f}, p={p_gnra:.4e}\n")
                f.write(f"  Mean saliency with GNRA tetraloops: {np.mean(gnra_groups[1]):.4f}\n")
                f.write(f"  Mean saliency without GNRA tetraloops: {np.mean(gnra_groups[0]):.4f}\n")
            else:
                f.write("GNRA Tetraloops: Insufficient data for t-test\n")
    
    except Exception as e:
        print(f"Error in RNA property statistical tests: {e}")
    
    return {
        'data': rna_df
    }