import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import os
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

# Helper: load CSV data if it exists, otherwise save the new data.
def get_or_save_csv(csv_filename, df, input_dir, output_dir):
    if input_dir is not None:
        in_path = os.path.join(input_dir, csv_filename)
        if os.path.exists(in_path):
            return pd.read_csv(in_path)
    out_path = os.path.join(output_dir, csv_filename)
    df.to_csv(out_path, index=False)
    return df

def iterate_over_att_and_prop(att_map, prop_map):
    """
    Given an attention map (numpy array) and its corresponding property map,
    yield pairs (single_att, single_prop) for each sample.
    
    - If att_map has shape [num_layers, num_heads, seq_length, seq_length] then yield it once.
    - If att_map has shape [B, num_layers, num_heads, seq_length, seq_length] then yield each
      sample. For the property map, if it is batched (ndim == att_map.ndim-1) then use the corresponding sample;
      otherwise, use the same property map for all samples.
    """
    if att_map.ndim == 4:
        yield att_map, prop_map
    elif att_map.ndim == 5:
        B = att_map.shape[0]
        # If property map is batched, expect its ndim == 3 (for 1D case: [B, seq_length] or for 2D: [B, seq_length, seq_length]).
        if prop_map.ndim == 3 or prop_map.ndim == 2:
            for b in range(B):
                yield att_map[b], prop_map[b] if prop_map.ndim == 3 else prop_map[b]
        else:
            # Not batched property: yield the same prop_map for every sample.
            for b in range(B):
                yield att_map[b], prop_map
    else:
        raise ValueError("Attention map must have 4 or 5 dimensions.")

###############################################################################
# THRESHOLD-BASED HEATMAPS
###############################################################################

def plot_attention_property_heatmap_for_threshold(attention_maps, properties, property_name, theta, output_dir, input_dir=None):
    """
    Create a heatmap (x: head, y: layer) for a specified theta.
    
    For each attention map (shape: [num_layers, num_heads, seq_length, seq_length]),
    for each layer and head, we determine the "high-confidence" entries as those 
    with attention > (theta * max(attention)) for that head. For binary properties 
    (with unique values {0,1}), we compute the fraction of these entries with value 1;
    For each attention map (shape: either [num_layers, num_heads, seq_length, seq_length] or
    [B, num_layers, num_heads, seq_length, seq_length]), for each layer and head, we determine the “high‐confidence”
    entries as those with attention > (theta * max(attention)) for that layer/head.
    
    For binary properties (values exactly {0,1}), we compute the fraction of these entries with value 1;
    for continuous properties we compute the average property value.
    
    The results are averaged over sequences (and batch samples) and saved as CSV and PNG.
    """
    # We assume that all attention maps share the same [num_layers, num_heads, ...] shape.
    sample_att = attention_maps[0].detach().cpu().numpy() if hasattr(attention_maps[0], "detach") else np.array(attention_maps[0])
    # In batched case, sample_att.shape is [B, num_layers, num_heads, seq_length, seq_length]
    if sample_att.ndim == 5:
        num_layers, num_heads = sample_att.shape[1:3]
    elif sample_att.ndim == 4:
        num_layers, num_heads = sample_att.shape[:2]
    else:
        raise ValueError("Attention maps must be 4D or 5D.")
    
    heatmap_vals = np.zeros((num_layers, num_heads))
    count_vals = np.zeros((num_layers, num_heads))
    
    for seq_idx in range(len(attention_maps)):
        att_tensor = attention_maps[seq_idx]
        att_tensor = torch.nn.functional.normalize(att_tensor, dim=-1)  # Normalize attention if needed.
        att_map = att_tensor.detach().cpu().numpy() if hasattr(att_tensor, "detach") else np.array(att_tensor)
        # print(att_map.shape)
        # Get the corresponding property map.
        # If the property map is 1D (shape: [seq_length,]), tile it to (seq_length, seq_length).
        # print(properties)
        # properties = properties[property_name]
        prop_map = properties[0][seq_idx]
        # print(prop_map.shape)
        if prop_map.ndim == 1:
            seq_length = att_map.shape[2]
            prop_mat = np.tile(prop_map[:, None], (1, seq_length))
        elif prop_map.ndim == 2:
            prop_mat = prop_map
        else:
            raise ValueError("Property map must be 1D or 2D.")
        # For each layer and head, compute the metric.
        for l in range(num_layers):
            for h in range(num_heads):
                att = att_map[l, h, :, :]
                max_val = np.max(att)
                if max_val == 0:
                    continue
                threshold_value = theta * max_val
                high_mask = (att > threshold_value)
                if np.sum(high_mask) == 0:
                    continue
                # Check if property is binary.
                unique_vals = np.unique(prop_mat)
                if (len(unique_vals) == 2) and (set(unique_vals.tolist()) <= {0, 1}):
                    metric = np.sum(prop_mat[high_mask] == 1) / np.sum(high_mask)
                else:
                    metric = np.mean(prop_mat[high_mask])
                heatmap_vals[l, h] += metric
                count_vals[l, h] += 1
    
    avg_heatmap = np.divide(heatmap_vals, count_vals, out=np.full_like(heatmap_vals, np.nan), where=(count_vals != 0))
    

    data_rows = []
    for l in range(num_layers):
        for h in range(num_heads):
            data_rows.append({
                "layer": l,
                "head": h,
                "average_metric": avg_heatmap[l, h],
                "count": count_vals[l, h]
            })
    df = pd.DataFrame(data_rows)
    csv_name = f"attention_property_heatmap_{property_name.replace(' ', '_')}_theta_{int(theta*100)}.csv"
    df = get_or_save_csv(csv_name, df, input_dir, output_dir)
    

    plt.figure(figsize=(6, 5))
    plt.imshow(avg_heatmap, aspect="auto", origin="lower", cmap=create_juelich_colormap())
    plt.colorbar(label="Average Metric")
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.title(f"Attention Property Heatmap: {property_name} (θ = {int(theta*100)}%)")
    plt.tight_layout()
    png_name = f"attention_property_heatmap_{property_name.replace(' ', '_')}_theta_{int(theta*100)}.svg"
    plt.savefig(os.path.join(output_dir, png_name), dpi=300, transparent=True, format='svg')
    plt.close()

def plot_attention_property_heatmaps_threshold(attention_maps, properties, property_names, output_dir, input_dir=None):
    """
    Iterate over theta thresholds (20%, 40%, 60%, 80%, 90%) and for each property
    call plot_attention_property_heatmap_for_threshold.
    """
    thresholds = [0.2, 0.4, 0.6, 0.8, 0.9, 0.92, 0.94, 0.96, 0.98, 0.99]
    for prop_name in property_names:
        for theta in thresholds:
            plot_attention_property_heatmap_for_threshold(attention_maps, properties, prop_name, theta, output_dir, input_dir)

###############################################################################
# TOP-L BASED HEATMAPS
###############################################################################

def plot_attention_property_heatmap_for_topL(attention_maps, properties, property_name, L, output_dir, input_dir=None):
    """
    Create a heatmap (x: head, y: layer) using the top L attention values per head.
    
    For each attention map (shape: [num_layers, num_heads, seq_length, seq_length] or with batch dim),
    for each layer and head, we flatten the attention matrix, select the top L values (if available),
    and compute the metric for the corresponding property values.
    
    For binary properties, this is the fraction of top L entries with value 1;
    for continuous properties, it is the average property value.
    
    The results are averaged over sequences (and batch samples) and saved as CSV and PNG.
    """
    sample_att = attention_maps[0].detach().cpu().numpy() if hasattr(attention_maps[0], "detach") else np.array(attention_maps[0])
    if sample_att.ndim == 5:
        num_layers, num_heads = sample_att.shape[1:3]
    elif sample_att.ndim == 4:
        num_layers, num_heads = sample_att.shape[:2]
    else:
        raise ValueError("Attention maps must be 4D or 5D.")
    
    heatmap_vals = np.zeros((num_layers, num_heads))
    count_vals = np.zeros((num_layers, num_heads))
    
    for seq_idx in range(len(attention_maps)):
        att_tensor = attention_maps[seq_idx]
        att_tensor = torch.nn.functional.normalize(att_tensor, dim=-1)  # Normalize attention if needed.
        att_map = att_tensor.detach().cpu().numpy() if hasattr(att_tensor, "detach") else np.array(att_tensor)
        # prop_map = properties[seq_idx]
        prop_map = properties[0][seq_idx]
        if prop_map.ndim == 1:
            seq_length = att_map.shape[2]
            prop_mat = np.tile(prop_map[:, None], (1, seq_length))
        elif prop_map.ndim == 2:
            prop_mat = prop_map
        else:
            raise ValueError("Property map must be 1D or 2D.")
        
        for l in range(num_layers):
            for h in range(num_heads):
                att = att_map[l, h, :, :]

                flat_att = att.flatten()
                if L > flat_att.size:
                    top_L_indices = np.argsort(flat_att)[::-1]  # take all sorted in descending order.
                else:
                    top_L_indices = np.argsort(flat_att)[-L:][::-1]

                top_rows, top_cols = np.unravel_index(top_L_indices, att.shape)

                top_props = prop_mat[top_rows, top_cols]

                unique_vals = np.unique(prop_mat)
                if (len(unique_vals) == 2) and (set(unique_vals.tolist()) <= {0, 1}):
                    metric = np.sum(top_props == 1) / len(top_props)
                else:
                    metric = np.mean(top_props)
                heatmap_vals[l, h] += metric
                count_vals[l, h] += 1
    
    avg_heatmap = np.divide(heatmap_vals, count_vals, out=np.full_like(heatmap_vals, np.nan), where=(count_vals != 0))
    
    data_rows = []
    for l in range(num_layers):
        for h in range(num_heads):
            data_rows.append({
                "layer": l,
                "head": h,
                "average_metric": avg_heatmap[l, h],
                "count": count_vals[l, h]
            })
    df = pd.DataFrame(data_rows)
    csv_name = f"attention_property_heatmap_{property_name.replace(' ', '_')}_topL_{L}.csv"
    df = get_or_save_csv(csv_name, df, input_dir, output_dir)


    plt.figure(figsize=(6, 5))
    plt.imshow(avg_heatmap, aspect="auto", origin="lower", cmap=create_juelich_colormap())
    plt.colorbar(label="Average Metric")
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.title(f"Attention Property Heatmap: {property_name} (Top {L})")
    plt.tight_layout()
    png_name = f"attention_property_heatmap_{property_name.replace(' ', '_')}_topL_{L}.svg"
    plt.savefig(os.path.join(output_dir, png_name), dpi=300, transparent=True, format='svg')
    plt.close()

def plot_attention_property_heatmaps_topL(attention_maps, properties, property_names, output_dir, input_dir=None):
    """
    Iterate over L values from 1 to 10 and for each property call plot_attention_property_heatmap_for_topL.
    """
    for prop_name in property_names:
        for L in range(1, 11):
            plot_attention_property_heatmap_for_topL(attention_maps, properties, prop_name, L, output_dir, input_dir)

###############################################################################
# MASTER FUNCTION & CLI
###############################################################################

def run_all_attention_plots(attention_maps, properties, property_names, output_dir, input_dir=None):
    """
    Run both the threshold-based and top-L based analyses.
    """
    plot_attention_property_heatmaps_threshold(attention_maps, properties, property_names, output_dir, input_dir)
    plot_attention_property_heatmaps_topL(attention_maps, properties, property_names, output_dir, input_dir)

def main():
    parser = argparse.ArgumentParser(
        description="Visualize how attention maps depend on nucleotide property maps."
    )
    parser.add_argument(
        "--attention_maps",
        type=str,
        required=True,
        help="Path to file containing a list of attention maps (torch tensors).",
    )
    parser.add_argument(
        "--properties",
        type=str,
        required=True,
        help="Path to file containing a list of property maps (numpy arrays).",
    )
    parser.add_argument(
        "--property_names",
        type=str,
        required=True,
        help="Comma-separated list of property names.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to store the generated plots and CSV files.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="(Optional) Directory from which to read precomputed CSV files.",
    )
    args = parser.parse_args()

    attention_maps = torch.load(args.attention_maps)
    properties = np.load(args.properties, allow_pickle=True)
    property_names = [name.strip() for name in args.property_names.split(",")]

    os.makedirs(args.output_dir, exist_ok=True)
    if args.input_dir is not None:
        os.makedirs(args.input_dir, exist_ok=True)
    
    run_all_attention_plots(attention_maps, properties, property_names, args.output_dir, args.input_dir)

if __name__ == "__main__":
    main()