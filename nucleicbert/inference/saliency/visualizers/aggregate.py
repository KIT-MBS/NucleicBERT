import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pandas as pd
import os
from typing import List, Dict, Optional, Tuple, Union
from pandas.plotting import scatter_matrix
from itertools import combinations

def aggregate_saliency_maps(
    saliency_maps: List[torch.Tensor], 
    save_dir: str,
    prefix: str = "aggregate"
) -> Dict:
    """
    Generate aggregate statistics for a list of saliency maps
    
    Args:
        saliency_maps: List of saliency map tensors
        save_dir: Directory to save the visualizations
        prefix: Prefix for saved files
        
    Returns:
        Dictionary of aggregate statistics
    """
    os.makedirs(save_dir, exist_ok=True)
    

    sal_maps_np = []
    for sal in saliency_maps:
        if hasattr(sal, 'detach'):
            sal_maps_np.append(sal.squeeze().detach().cpu().numpy())
        else:
            sal_maps_np.append(np.array(sal).squeeze())
    

    mean_saliency = np.mean([np.mean(sal) for sal in sal_maps_np])
    std_saliency = np.std([np.mean(sal) for sal in sal_maps_np])
    max_saliency = np.max([np.max(sal) for sal in sal_maps_np])
    min_saliency = np.min([np.min(sal) for sal in sal_maps_np])
    

    all_saliency_values = np.concatenate([sal.flatten() for sal in sal_maps_np])
    
    
    # 1. Histogram of all saliency values
    plt.figure(figsize=(10, 6))
    plt.hist(all_saliency_values, bins=50, alpha=0.7, color='darkblue')
    plt.axvline(mean_saliency, color='red', linestyle='--', label=f'Mean: {mean_saliency:.4f}')
    plt.xlabel('Saliency Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of All Saliency Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, f"{prefix}_saliency_histogram.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Box plot of saliency distributions per sample
    plt.figure(figsize=(12, 6))
    plt.boxplot([sal.flatten() for sal in sal_maps_np], labels=[f'Sample {i+1}' for i in range(len(sal_maps_np))])
    plt.xlabel('Sample')
    plt.ylabel('Saliency Value')
    plt.title('Saliency Distribution Across Samples')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_saliency_boxplot.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Line plot of mean/max saliency per sample
    plt.figure(figsize=(12, 6))
    samples = np.arange(1, len(sal_maps_np) + 1)
    mean_vals = [np.mean(sal) for sal in sal_maps_np]
    max_vals = [np.max(sal) for sal in sal_maps_np]
    plt.plot(samples, mean_vals, 'o-', color='blue', label='Mean Saliency')
    plt.plot(samples, max_vals, 'o-', color='red', label='Max Saliency')
    plt.xlabel('Sample')
    plt.ylabel('Saliency Value')
    plt.title('Mean and Max Saliency per Sample')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(samples)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_saliency_per_sample.png"), dpi=300, bbox_inches='tight')
    plt.close()
    

    stats = {
        'mean_saliency': mean_saliency,
        'std_saliency': std_saliency,
        'max_saliency': max_saliency,
        'min_saliency': min_saliency,
        'sample_means': mean_vals,
        'sample_maxes': max_vals
    }
    

    with open(os.path.join(save_dir, f"{prefix}_saliency_stats.txt"), 'w') as f:
        f.write("Aggregate Saliency Statistics\n")
        f.write("===========================\n\n")
        f.write(f"Number of samples: {len(sal_maps_np)}\n")
        f.write(f"Mean saliency: {mean_saliency:.6f}\n")
        f.write(f"Standard deviation: {std_saliency:.6f}\n")
        f.write(f"Maximum saliency: {max_saliency:.6f}\n")
        f.write(f"Minimum saliency: {min_saliency:.6f}\n\n")
        
        f.write("Per-sample statistics:\n")
        for i, (mean_val, max_val) in enumerate(zip(mean_vals, max_vals)):
            f.write(f"Sample {i+1}: Mean = {mean_val:.6f}, Max = {max_val:.6f}\n")
    
    return stats

def analyze_saliency_vs_contact_properties(
    saliency_maps: List[torch.Tensor],
    contact_maps: List[torch.Tensor],
    save_dir: str,
    prefix: str = "aggregate"
) -> Dict:
    """
    Analyze the relationship between saliency and various contact map properties
    across multiple samples
    
    Args:
        saliency_maps: List of saliency map tensors
        contact_maps: List of contact map tensors
        save_dir: Directory to save the visualizations
        prefix: Prefix for saved files
        
    Returns:
        Dictionary of analysis results
    """

    os.makedirs(save_dir, exist_ok=True)
    

    sal_maps_np = []
    contact_maps_np = []
    
    for sal, contact in zip(saliency_maps, contact_maps):
        if hasattr(sal, 'detach'):
            sal_maps_np.append(sal.squeeze().detach().cpu().numpy())
        else:
            sal_maps_np.append(np.array(sal).squeeze())
            
        if hasattr(contact, 'detach'):
            contact_maps_np.append(contact.squeeze().detach().cpu().numpy())
        else:
            contact_maps_np.append(np.array(contact).squeeze())
    

    property_data = []
    
    for i, (sal, contact) in enumerate(zip(sal_maps_np, contact_maps_np)):

        seq_len = len(sal)
        if contact.shape[0] > seq_len:
            contact = contact[:seq_len, :seq_len]
        elif contact.shape[0] < seq_len:
            # Pad contact map if it's smaller
            pad_size = seq_len - contact.shape[0]
            contact = np.pad(contact, ((0, pad_size), (0, pad_size)), 'constant')
        

        contact_degree = np.sum(contact, axis=1)  
        
        long_range_threshold = 24
        long_range_contacts = np.zeros(seq_len)
        short_range_contacts = np.zeros(seq_len)
        
        for pos in range(seq_len):
            for j in range(seq_len):
                if contact[pos, j] == 1:  # Binary contact
                    if abs(pos - j) > long_range_threshold:
                        long_range_contacts[pos] += 1
                    else:
                        short_range_contacts[pos] += 1
        

        long_range_ratio = np.zeros(seq_len)
        has_contacts = (short_range_contacts + long_range_contacts) > 0
        long_range_ratio[has_contacts] = long_range_contacts[has_contacts] / (short_range_contacts[has_contacts] + long_range_contacts[has_contacts])
        

        for pos in range(seq_len):
            property_data.append({
                'sample': i,
                'position': pos,
                'saliency': sal[pos],
                'contact_degree': contact_degree[pos],
                'long_range_ratio': long_range_ratio[pos],
                'is_long_range': long_range_contacts[pos] > 0,
                'is_short_range': short_range_contacts[pos] > 0,
                'has_contact': contact_degree[pos] > 0
            })
    
    df = pd.DataFrame(property_data)
    
    df.to_csv(os.path.join(save_dir, f"{prefix}_saliency_contact_data.csv"), index=False)
    
    
    # 1. Box plot comparing saliency for positions with and without contacts
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='has_contact', y='saliency', data=df)
    plt.xlabel('Has Contact')
    plt.ylabel('Saliency')
    plt.title('Saliency Distribution: Contact vs. No Contact')
    plt.xticks([0, 1], ['No', 'Yes'])
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_saliency_by_contact.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Box plot comparing saliency for positions with long-range vs. short-range contacts
    has_contacts_df = df[df['has_contact']]
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='is_long_range', y='saliency', data=has_contacts_df)
    plt.xlabel('Has Long-Range Contact')
    plt.ylabel('Saliency')
    plt.title('Saliency Distribution: Long-Range vs. Short-Range Contacts')
    plt.xticks([0, 1], ['No', 'Yes'])
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_saliency_by_long_range.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Scatter plot of saliency vs. contact degree
    plt.figure(figsize=(10, 6))
    sns.regplot(x='contact_degree', y='saliency', data=df, scatter_kws={'alpha': 0.3})
    plt.xlabel('Contact Degree')
    plt.ylabel('Saliency')
    plt.title('Saliency vs. Contact Degree')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_saliency_vs_degree.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Scatter plot of saliency vs. long-range ratio (only for positions with contacts)
    plt.figure(figsize=(10, 6))
    sns.regplot(x='long_range_ratio', y='saliency', data=has_contacts_df, scatter_kws={'alpha': 0.3})
    plt.xlabel('Long-Range Ratio')
    plt.ylabel('Saliency')
    plt.title('Saliency vs. Long-Range Ratio')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_saliency_vs_long_range_ratio.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Correlation heatmap
    corr_cols = ['saliency', 'contact_degree', 'long_range_ratio']
    corr_df = df[corr_cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
    plt.title('Correlation Between Saliency and Contact Properties')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_correlation_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Group bar chart showing average saliency by contact properties
    # Get average saliency for different groups
    means = {
        'No Contact': df[~df['has_contact']]['saliency'].mean(),
        'Short-Range Only': df[df['is_short_range'] & ~df['is_long_range']]['saliency'].mean(),
        'Long-Range': df[df['is_long_range']]['saliency'].mean()
    }
    stds = {
        'No Contact': df[~df['has_contact']]['saliency'].std(),
        'Short-Range Only': df[df['is_short_range'] & ~df['is_long_range']]['saliency'].std(),
        'Long-Range': df[df['is_long_range']]['saliency'].std()
    }
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(means.keys(), means.values(), yerr=list(stds.values()), capsize=10)
    plt.xlabel('Contact Type')
    plt.ylabel('Average Saliency')
    plt.title('Average Saliency by Contact Type')
    plt.grid(True, alpha=0.3, axis='y')
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01, 
                f'{list(means.values())[i]:.3f}', 
                ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_avg_saliency_by_contact_type.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    try:
        from scipy import stats
        
        # T-test between contact vs. no contact
        t_stat_contact, p_val_contact = stats.ttest_ind(
            df[df['has_contact']]['saliency'], 
            df[~df['has_contact']]['saliency'], 
            equal_var=False
        )
        
        # T-test between long-range vs. short-range only
        has_long = df[df['is_long_range']]['saliency']
        short_only = df[df['is_short_range'] & ~df['is_long_range']]['saliency']
        if len(has_long) > 0 and len(short_only) > 0:
            t_stat_range, p_val_range = stats.ttest_ind(
                has_long, 
                short_only, 
                equal_var=False
            )
        else:
            t_stat_range, p_val_range = np.nan, np.nan
        
        # Correlation tests
        corr_degree, p_degree = stats.pearsonr(df['contact_degree'], df['saliency'])
        
        # Only do long-range ratio correlation if we have data
        if len(has_contacts_df) > 0:
            corr_long_range, p_long_range = stats.pearsonr(
                has_contacts_df['long_range_ratio'], 
                has_contacts_df['saliency']
            )
        else:
            corr_long_range, p_long_range = np.nan, np.nan
        
        # Save statistical results
        with open(os.path.join(save_dir, f"{prefix}_statistical_tests.txt"), 'w') as f:
            f.write("Statistical Tests for Saliency vs. Contact Properties\n")
            f.write("==================================================\n\n")
            
            f.write("T-tests:\n")
            f.write(f"Contact vs. No Contact: t={t_stat_contact:.4f}, p={p_val_contact:.4e}\n")
            if not np.isnan(t_stat_range):
                f.write(f"Long-Range vs. Short-Range Only: t={t_stat_range:.4f}, p={p_val_range:.4e}\n\n")
            else:
                f.write("Long-Range vs. Short-Range Only: Insufficient data\n\n")
            
            f.write("Correlations:\n")
            f.write(f"Saliency vs. Contact Degree: r={corr_degree:.4f}, p={p_degree:.4e}\n")
            if not np.isnan(corr_long_range):
                f.write(f"Saliency vs. Long-Range Ratio: r={corr_long_range:.4f}, p={p_long_range:.4e}\n")
            else:
                f.write("Saliency vs. Long-Range Ratio: Insufficient data\n")
        
        # Return the statistical results
        stats_results = {
            't_contact': t_stat_contact,
            'p_contact': p_val_contact,
            't_range': t_stat_range,
            'p_range': p_val_range,
            'corr_degree': corr_degree,
            'p_degree': p_degree,
            'corr_long_range': corr_long_range,
            'p_long_range': p_long_range
        }
    except ImportError:
        # If scipy isn't available, return empty stats
        stats_results = {}
    
    return {
        'data': df,
        'stats': stats_results,
        'means': means,
        'stds': stds
    }

def plot_normalized_saliency_threshold(
    saliency_maps: List[torch.Tensor],
    contact_maps: List[torch.Tensor],
    save_dir: str,
    prefix: str = "aggregate"
) -> None:
    """
    Analyze the relationship between normalized saliency thresholds and contact properties
    
    Args:
        saliency_maps: List of saliency map tensors
        contact_maps: List of contact map tensors
        save_dir: Directory to save the visualizations
        prefix: Prefix for saved files
    """
    os.makedirs(save_dir, exist_ok=True)
    
    sal_maps_np = []
    contact_maps_np = []
    
    for sal, contact in zip(saliency_maps, contact_maps):
        if hasattr(sal, 'detach'):
            sal_maps_np.append(sal.squeeze().detach().cpu().numpy())
        else:
            sal_maps_np.append(np.array(sal).squeeze())
            
        if hasattr(contact, 'detach'):
            contact_maps_np.append(contact.squeeze().detach().cpu().numpy())
        else:
            contact_maps_np.append(np.array(contact).squeeze())
    
    norm_sal_maps = []
    for sal in sal_maps_np:
        max_val = np.max(sal)
        if max_val > 0:
            norm_sal_maps.append(sal / max_val)
        else:
            norm_sal_maps.append(sal)

    has_contact = []
    is_long_range = []
    norm_saliency = []
    
    long_range_threshold = 24
    
    for sal, contact in zip(norm_sal_maps, contact_maps_np):

        seq_len = len(sal)
        if contact.shape[0] > seq_len:
            contact = contact[:seq_len, :seq_len]
        elif contact.shape[0] < seq_len:
            pad_size = seq_len - contact.shape[0]
            contact = np.pad(contact, ((0, pad_size), (0, pad_size)), 'constant')
        

        for pos in range(seq_len):
            contacts = []
            for j in range(seq_len):
                if contact[pos, j] == 1:
                    contacts.append(j)
            
            has_contact.append(len(contacts) > 0)
            has_long_range = any(abs(pos - j) > long_range_threshold for j in contacts)
            is_long_range.append(has_long_range)
            norm_saliency.append(sal[pos])
    

    thresholds = np.linspace(0, 1, 101)  
    

    contact_fractions = []
    no_contact_fractions = []
    long_range_fractions = []
    short_range_fractions = []
    
    for threshold in thresholds:

        high_saliency = np.array(norm_saliency) >= threshold
        

        with_contacts = np.array(has_contact)
        without_contacts = ~with_contacts
        

        with_long_range = np.array(is_long_range)
        with_short_range_only = with_contacts & ~with_long_range
        

        if sum(with_contacts) > 0:
            contact_fractions.append(sum(high_saliency & with_contacts) / sum(with_contacts))
        else:
            contact_fractions.append(0)
            
        if sum(without_contacts) > 0:
            no_contact_fractions.append(sum(high_saliency & without_contacts) / sum(without_contacts))
        else:
            no_contact_fractions.append(0)
            
        if sum(with_long_range) > 0:
            long_range_fractions.append(sum(high_saliency & with_long_range) / sum(with_long_range))
        else:
            long_range_fractions.append(0)
            
        if sum(with_short_range_only) > 0:
            short_range_fractions.append(sum(high_saliency & with_short_range_only) / sum(with_short_range_only))
        else:
            short_range_fractions.append(0)
    

    plt.figure(figsize=(12, 8))
    plt.plot(thresholds, contact_fractions, 'b-', label='With Contacts')
    plt.plot(thresholds, no_contact_fractions, 'r-', label='Without Contacts')
    plt.plot(thresholds, long_range_fractions, 'g-', label='With Long-Range Contacts')
    plt.plot(thresholds, short_range_fractions, 'c-', label='With Short-Range Contacts Only')
    
    plt.xlabel('Normalized Saliency Threshold')
    plt.ylabel('Fraction of Positions')
    plt.title('Fraction of Positions Above Saliency Threshold by Contact Type')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_saliency_threshold_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()
    

    threshold_data = pd.DataFrame({
        'threshold': thresholds,
        'with_contacts': contact_fractions,
        'without_contacts': no_contact_fractions,
        'with_long_range': long_range_fractions,
        'with_short_range_only': short_range_fractions
    })
    
    threshold_data.to_csv(os.path.join(save_dir, f"{prefix}_saliency_threshold_data.csv"), index=False)

def run_all_aggregate_analyses(
    saliency_maps: List[torch.Tensor],
    contact_maps: List[torch.Tensor],
    save_dir: str,
    prefix: str = "aggregate"
) -> Dict:
    """
    Run all aggregate analyses on a set of saliency maps and contact maps
    
    Args:
        saliency_maps: List of saliency map tensors
        contact_maps: List of contact map tensors
        save_dir: Directory to save all visualizations
        prefix: Prefix for saved files
        
    Returns:
        Dictionary with all analysis results
    """
    os.makedirs(save_dir, exist_ok=True)
    

    saliency_stats = aggregate_saliency_maps(saliency_maps, save_dir, prefix)
    

    contact_analysis = analyze_saliency_vs_contact_properties(saliency_maps, contact_maps, save_dir, prefix)
    

    plot_normalized_saliency_threshold(saliency_maps, contact_maps, save_dir, prefix)
    
    # Create index.html to navigate all plots
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Saliency Analysis Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333; }}
            .plot-container {{ margin: 20px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}
            img {{ max-width: 100%; height: auto; }}
            .summary {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>RNA Saliency Analysis Results</h1>
        
        <div class="summary">
            <h2>Summary Statistics</h2>
            <p>
                Number of Samples: {len(saliency_maps)}<br>
                Mean Saliency: {saliency_stats['mean_saliency']:.4f}<br>
                Standard Deviation: {saliency_stats['std_saliency']:.4f}<br>
            </p>
        </div>
        
        <h2>Saliency Distribution Analysis</h2>
        
        <div class="plot-container">
            <h3>Distribution of All Saliency Values</h3>
            <img src="{prefix}_saliency_histogram.png" alt="Saliency Histogram">
            <p>Histogram showing the distribution of all saliency values across all samples.</p>
        </div>
        
        <div class="plot-container">
            <h3>Saliency Distribution Across Samples</h3>
            <img src="{prefix}_saliency_boxplot.png" alt="Saliency Boxplot">
            <p>Box plot comparing saliency distributions for each sample.</p>
        </div>
        
        <div class="plot-container">
            <h3>Mean and Max Saliency per Sample</h3>
            <img src="{prefix}_saliency_per_sample.png" alt="Saliency per Sample">
            <p>Line plot showing the mean and maximum saliency values for each sample.</p>
        </div>
        
        <h2>Contact Property Analysis</h2>
        
        <div class="plot-container">
            <h3>Saliency Distribution: Contact vs. No Contact</h3>
            <img src="{prefix}_saliency_by_contact.png" alt="Saliency by Contact">
            <p>Box plot comparing saliency distributions for positions with and without contacts.</p>
        </div>
        
        <div class="plot-container">
            <h3>Saliency Distribution: Long-Range vs. Short-Range Contacts</h3>
            <img src="{prefix}_saliency_by_long_range.png" alt="Saliency by Long Range">
            <p>Box plot comparing saliency distributions for positions with long-range versus short-range contacts.</p>
        </div>
        
        <div class="plot-container">
            <h3>Saliency vs. Contact Degree</h3>
            <img src="{prefix}_saliency_vs_degree.png" alt="Saliency vs Degree">
            <p>Scatter plot with regression line showing relationship between contact degree and saliency.</p>
        </div>
        
        <div class="plot-container">
            <h3>Saliency vs. Long-Range Ratio</h3>
            <img src="{prefix}_saliency_vs_long_range_ratio.png" alt="Saliency vs Long Range Ratio">
            <p>Scatter plot with regression line showing relationship between long-range ratio and saliency.</p>
        </div>
        
        <div class="plot-container">
            <h3>Correlation Heatmap</h3>
            <img src="{prefix}_correlation_heatmap.png" alt="Correlation Heatmap">
            <p>Heatmap showing correlations between saliency and various contact properties.</p>
        </div>
        
        <div class="plot-container">
            <h3>Average Saliency by Contact Type</h3>
            <img src="{prefix}_avg_saliency_by_contact_type.png" alt="Average Saliency by Contact Type">
            <p>Bar chart showing the average saliency for different types of contacts.</p>
        </div>
        
        <h2>Threshold Analysis</h2>
        
        <div class="plot-container">
            <h3>Fraction of Positions Above Saliency Threshold</h3>
            <img src="{prefix}_saliency_threshold_analysis.png" alt="Saliency Threshold Analysis">
            <p>Line plot showing how different contact properties relate to saliency thresholds.</p>
        </div>
        
        <footer>
            <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </footer>
    </body>
    </html>
    """
    
    # Save HTML file
    with open(os.path.join(save_dir, f"{prefix}_analysis_index.html"), 'w') as f:
        f.write(html_content)
    
    return {
        'saliency_stats': saliency_stats,
        'contact_analysis': contact_analysis
    }