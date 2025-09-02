import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
from itertools import combinations


def get_or_save_csv(csv_filename, df, input_dir, output_dir):
    if input_dir is not None:
        in_path = os.path.join(input_dir, csv_filename)
        if os.path.exists(in_path):
            return pd.read_csv(in_path)
    out_path = os.path.join(output_dir, csv_filename)
    df.to_csv(out_path, index=False)
    return df

def plot_scatter_continuous(saliency_maps, properties, property_names, output_dir, input_dir=None):
    """
    For each property that is continuous, plot a scatter plot (property vs. saliency)
    with a linear regression line. Also save the underlying data as CSV.
    """
    if properties[0].ndim == 1:
        n_props = 1
    else:
        n_props = properties[0].shape[0]
    for j in range(n_props):
        saliency_all = []
        prop_all = []
        for i in range(len(saliency_maps)):
            sal = saliency_maps[i].detach().cpu().numpy() if hasattr(saliency_maps[i], "detach") else np.array(saliency_maps[i])
            if properties[i].ndim == 1:
                p = properties[i]
            else:
                p = properties[i][j, :]
            saliency_all.append(sal)
            prop_all.append(p)
        saliency_all = np.concatenate(saliency_all)
        prop_all = np.concatenate(prop_all)

        unique_vals = np.unique(prop_all)
        if not (len(unique_vals) == 2 and set(unique_vals).issubset({0, 1})):
            df = pd.DataFrame({"property": prop_all, "saliency": saliency_all})
            csv_name = f"scatter_continuous_{property_names[j].replace(' ', '_')}.csv"
            df = get_or_save_csv(csv_name, df, input_dir, output_dir)

            plt.figure()
            plt.scatter(df["property"], df["saliency"], alpha=0.5)
            plt.xlabel(property_names[j])
            plt.ylabel("Saliency")
            plt.title(f"Scatter Plot: Saliency vs. {property_names[j]}")

            if len(df) > 1:
                slope, intercept = np.polyfit(df["property"], df["saliency"], 1)
                x_vals = np.linspace(df["property"].min(), df["property"].max(), 100)
                y_vals = slope * x_vals + intercept
                plt.plot(x_vals, y_vals, color='red', label=f'Fit: y={slope:.2f}x+{intercept:.2f}')
                plt.legend()
            plt.tight_layout()
            png_name = f"scatter_continuous_{property_names[j].replace(' ', '_')}.png"
            plt.savefig(os.path.join(output_dir, png_name))
            plt.close()

def plot_boxplot_binary(saliency_maps, properties, property_names, output_dir, input_dir=None):
    """
    For each binary property, produce a box plot comparing the saliency distributions
    for nucleotides with property value 0 vs. 1. Also save the underlying data as CSV.
    """
    if properties[0].ndim == 1:
        n_props = 1
    else:
        n_props = properties[0].shape[0]
    for j in range(n_props):
        rows = []
        for i in range(len(saliency_maps)):
            sal = saliency_maps[i].detach().cpu().numpy() if hasattr(saliency_maps[i], "detach") else np.array(saliency_maps[i])
            if properties[i].ndim == 1:
                p = properties[i]
            else:
                p = properties[i][j, :]
            unique_vals = np.unique(p)
            if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1}):
                for group in [0, 1]:
                    indices = np.where(p == group)[0]
                    for val in sal[indices]:
                        rows.append({"group": group, "saliency": val})
        if rows:
            df = pd.DataFrame(rows)
            csv_name = f"boxplot_binary_{property_names[j].replace(' ', '_')}.csv"
            df = get_or_save_csv(csv_name, df, input_dir, output_dir)
            
            plt.figure()
            df.boxplot(column="saliency", by="group")
            plt.xlabel(property_names[j])
            plt.ylabel("Saliency")
            plt.title(f"Box Plot: Saliency by {property_names[j]}")
            plt.suptitle("")  # Remove automatic title
            plt.tight_layout()
            png_name = f"boxplot_binary_{property_names[j].replace(' ', '_')}.png"
            plt.savefig(os.path.join(output_dir, png_name))
            plt.close()

def plot_grouped_bar_chart(saliency_maps, properties, property_names, output_dir, input_dir=None):
    """
    For each binary property, compute the average (and standard deviation) of saliency
    for nucleotides with value 0 and 1, and plot these as grouped bars.
    Also save the computed summary as CSV.
    """
    if properties[0].ndim == 1:
        n_props = 1
    else:
        n_props = properties[0].shape[0]
    for j in range(n_props):
        group0_vals = []
        group1_vals = []
        for i in range(len(saliency_maps)):
            sal = saliency_maps[i].detach().cpu().numpy() if hasattr(saliency_maps[i], "detach") else np.array(saliency_maps[i])
            if properties[i].ndim == 1:
                p = properties[i]
            else:
                p = properties[i][j, :]
            unique_vals = np.unique(p)
            if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1}):
                group0_vals.append(sal[p == 0])
                group1_vals.append(sal[p == 1])
        if group0_vals and group1_vals:
            group0_vals = np.concatenate(group0_vals)
            group1_vals = np.concatenate(group1_vals)
            mean0, std0 = np.mean(group0_vals), np.std(group0_vals)
            mean1, std1 = np.mean(group1_vals), np.std(group1_vals)
            df = pd.DataFrame({
                "group": [0, 1],
                "mean": [mean0, mean1],
                "std": [std0, std1]
            })
            csv_name = f"grouped_bar_chart_{property_names[j].replace(' ', '_')}.csv"
            df = get_or_save_csv(csv_name, df, input_dir, output_dir)
            
            plt.figure()
            plt.bar([0, 1], [mean0, mean1], yerr=[std0, std1], tick_label=["0", "1"])
            plt.xlabel(property_names[j])
            plt.ylabel("Average Saliency")
            plt.title(f"Grouped Bar Chart: {property_names[j]}")
            plt.tight_layout()
            png_name = f"grouped_bar_chart_{property_names[j].replace(' ', '_')}.png"
            plt.savefig(os.path.join(output_dir, png_name))
            plt.close()

def plot_correlation_heatmap(saliency_maps, properties, property_names, output_dir, input_dir=None):
    """
    Combine data from all sequences into a DataFrame (with saliency and each property)
    and compute the correlation matrix. Save the correlation matrix as CSV and visualize it.
    """
    combined_data = {"Saliency": []}
    if properties[0].ndim == 1:
        combined_data[property_names[0]] = []
    else:
        n_props = properties[0].shape[0]
        for j in range(n_props):
            name = property_names[j] if j < len(property_names) else f"Property{j}"
            combined_data[name] = []
    for i in range(len(saliency_maps)):
        sal = saliency_maps[i].detach().cpu().numpy() if hasattr(saliency_maps[i], "detach") else np.array(saliency_maps[i])
        combined_data["Saliency"].extend(sal)
        if properties[i].ndim == 1:
            combined_data[property_names[0]].extend(properties[i])
        else:
            for j in range(properties[i].shape[0]):
                name = property_names[j] if j < len(property_names) else f"Property{j}"
                combined_data[name].extend(properties[i][j, :])
    df = pd.DataFrame(combined_data)
    corr = df.corr()
    csv_name = "correlation_heatmap.csv"
    corr.to_csv(os.path.join(output_dir, csv_name), index=True)  
    plt.figure()
    plt.imshow(corr, cmap="viridis", interpolation="none")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    png_name = "correlation_heatmap.png"
    plt.savefig(os.path.join(output_dir, png_name))
    plt.close()

def plot_overlay_on_sequence(saliency_maps, properties, property_names, output_dir, input_dir=None):
    """
    For each sequence, plot the saliency map as a line and overlay the positions
    where any binary property is true. Also save the underlying sequence data as CSV.
    """
    for i in range(len(saliency_maps)):
        sal = saliency_maps[i].detach().cpu().numpy() if hasattr(saliency_maps[i], "detach") else np.array(saliency_maps[i])
        x = np.arange(len(sal))
        df = pd.DataFrame({"position": x, "saliency": sal})
        if properties[i].ndim == 1:
            p = properties[i]
            unique_vals = np.unique(p)
            if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1}):
                df[property_names[0]] = p
        else:
            for j in range(properties[i].shape[0]):
                p = properties[i][j, :]
                unique_vals = np.unique(p)
                if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1}):
                    colname = property_names[j] if j < len(property_names) else f"Property{j}"
                    df[colname] = p
        csv_name = f"overlay_sequence_{i}.csv"
        df = get_or_save_csv(csv_name, df, input_dir, output_dir)

        plt.figure()
        plt.plot(x, sal, label="Saliency")
        if properties[i].ndim == 1:
            p = properties[i]
            if len(np.unique(p)) == 2 and set(np.unique(p)).issubset({0, 1}):
                indices = np.where(p == 1)[0]
                plt.scatter(indices, sal[indices], color="red", label=property_names[0])
        else:
            for j in range(properties[i].shape[0]):
                p = properties[i][j, :]
                if len(np.unique(p)) == 2 and set(np.unique(p)).issubset({0, 1}):
                    indices = np.where(p == 1)[0]
                    label = property_names[j] if j < len(property_names) else f"Property{j}"
                    plt.scatter(indices, sal[indices], label=label)
        plt.xlabel("Nucleotide Position")
        plt.ylabel("Saliency")
        plt.title(f"Overlay Plot: Sequence {i}")
        plt.legend()
        plt.tight_layout()
        png_name = f"overlay_sequence_{i}.png"
        plt.savefig(os.path.join(output_dir, png_name))
        plt.close()

def plot_composite_multi_panel(saliency_maps, properties, property_names, output_dir, input_dir=None):
    """
    Create a composite (3-panel) figure using the first sequence as an example:
      - Panel 1: Raw saliency map (line plot)
      - Panel 2: Annotation of a binary property (if available)
      - Panel 3: Scatter plot of saliency vs. continuous property (if available)
    Also save the underlying data as CSV.
    """
    if len(saliency_maps) == 0:
        return
    sal = saliency_maps[0].detach().cpu().numpy() if hasattr(saliency_maps[0], "detach") else np.array(saliency_maps[0])
    x = np.arange(len(sal))
    bin_prop = None
    bin_prop_name = None
    cont_prop = None
    cont_prop_name = None
    if properties[0].ndim == 1:
        p = properties[0]
        if len(np.unique(p)) == 2 and set(np.unique(p)).issubset({0, 1}):
            bin_prop = p
            bin_prop_name = property_names[0]
        else:
            cont_prop = p
            cont_prop_name = property_names[0]
    else:
        for j in range(properties[0].shape[0]):
            p = properties[0][j, :]
            if bin_prop is None and (len(np.unique(p)) == 2 and set(np.unique(p)).issubset({0, 1})):
                bin_prop = p
                bin_prop_name = property_names[j] if j < len(property_names) else f"Property{j}"
            elif cont_prop is None:
                cont_prop = p
                cont_prop_name = property_names[j] if j < len(property_names) else f"Property{j}"
    data = {"position": x, "saliency": sal}
    if bin_prop is not None:
        data[bin_prop_name] = bin_prop
    if cont_prop is not None:
        data[cont_prop_name] = cont_prop
    df = pd.DataFrame(data)
    csv_name = "composite_multi_panel.csv"
    df = get_or_save_csv(csv_name, df, input_dir, output_dir)

    fig, axs = plt.subplots(3, 1, figsize=(8, 12))
    axs[0].plot(x, sal)
    axs[0].set_title("Raw Saliency Map")
    axs[0].set_xlabel("Nucleotide Position")
    axs[0].set_ylabel("Saliency")
    if bin_prop is not None:
        axs[1].plot(x, bin_prop, 'o', markersize=4)
        axs[1].set_title(f"Binary Property: {bin_prop_name}")
        axs[1].set_xlabel("Nucleotide Position")
        axs[1].set_ylabel(bin_prop_name)
    else:
        axs[1].axis('off')
        axs[1].set_title("No binary property available")
    if cont_prop is not None:
        axs[2].scatter(cont_prop, sal, alpha=0.5)
        if len(cont_prop) > 1:
            slope, intercept = np.polyfit(cont_prop, sal, 1)
            x_vals = np.linspace(np.min(cont_prop), np.max(cont_prop), 100)
            y_vals = slope * x_vals + intercept
            axs[2].plot(x_vals, y_vals, color='red', label=f'Fit: y={slope:.2f}x+{intercept:.2f}')
            axs[2].legend()
        axs[2].set_title(f"Saliency vs. Continuous Property: {cont_prop_name}")
        axs[2].set_xlabel(cont_prop_name)
        axs[2].set_ylabel("Saliency")
    else:
        axs[2].axis('off')
        axs[2].set_title("No continuous property available")
    plt.tight_layout()
    png_name = "composite_multi_panel.png"
    plt.savefig(os.path.join(output_dir, png_name))
    plt.close()

def plot_pairwise_scatter_matrix(saliency_maps, properties, property_names, output_dir, input_dir=None):
    """
    If more than one property is available (i.e. properties are 2D), then for every pair
    of properties, combine saliency and the two property columns across all sequences,
    save the data as CSV, and plot a scatter matrix.
    """
    if properties[0].ndim == 1:
        return
    n_props = properties[0].shape[0]
    for idx1, idx2 in combinations(range(n_props), 2):
        combined = {"Saliency": [], 
                    property_names[idx1] if idx1 < len(property_names) else f"Property{idx1}": [],
                    property_names[idx2] if idx2 < len(property_names) else f"Property{idx2}": []}
        for i in range(len(saliency_maps)):
            sal = saliency_maps[i].detach().cpu().numpy() if hasattr(saliency_maps[i], "detach") else np.array(saliency_maps[i])
            p1 = properties[i][idx1, :]
            p2 = properties[i][idx2, :]
            combined["Saliency"].extend(sal)
            col1 = property_names[idx1] if idx1 < len(property_names) else f"Property{idx1}"
            col2 = property_names[idx2] if idx2 < len(property_names) else f"Property{idx2}"
            combined[col1].extend(p1)
            combined[col2].extend(p2)
        df = pd.DataFrame(combined)
        csv_name = f"pairwise_scatter_matrix_{property_names[idx1].replace(' ', '_')}_{property_names[idx2].replace(' ', '_')}.csv"
        df = get_or_save_csv(csv_name, df, input_dir, output_dir)
        
        fig = scatter_matrix(df, figsize=(10, 10), diagonal='kde')
        plt.suptitle(f"Pairwise Scatter Matrix: {property_names[idx1]} & {property_names[idx2]}")
        plt.tight_layout()
        png_name = f"pairwise_scatter_matrix_{property_names[idx1].replace(' ', '_')}_{property_names[idx2].replace(' ', '_')}.png"
        plt.savefig(os.path.join(output_dir, png_name))
        plt.close()

def plot_normalized_saliency_threshold(saliency_maps, properties, property_names, output_dir, input_dir=None):
    """
    For each property, compute the percentage of nucleotides that (a) have the given property,
    and (b) have normalized saliency over a threshold.
    
    For binary properties, vary saliency threshold (theta) from 0 to 1 and compute:
      percentage = (# nucleotides with property==1 and saliency > theta) / (# nucleotides with property==1) * 100
      
    For continuous (float) properties, vary both a property threshold (to binarize the property)
    and a saliency threshold, and plot the result as a 2D heatmap.
    
    The underlying data is saved as CSV.
    """
    if properties[0].ndim == 1:
        n_props = 1
    else:
        n_props = properties[0].shape[0]
    
    for j in range(n_props):
        prop_list = []
        sal_list = []
        for i in range(len(saliency_maps)):
            sal = saliency_maps[i].detach().cpu().numpy() if hasattr(saliency_maps[i], "detach") else np.array(saliency_maps[i])
            if properties[i].ndim == 1:
                p = properties[i]
            else:
                p = properties[i][j, :]
            prop_list.append(p)
            sal_list.append(sal)
        all_prop = np.concatenate(prop_list)
        unique_vals = np.unique(all_prop)
        if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1}):
            thresholds = np.linspace(0, 1, 50)
            percents = []
            for theta in thresholds:
                perc_seq = []
                for p, sal in zip(prop_list, sal_list):
                    max_sal = np.max(sal) if np.max(sal) > 0 else 1
                    sal_norm = sal / max_sal
                    mask = (p == 1)
                    if np.sum(mask) > 0:
                        percent = np.sum(sal_norm[mask] > theta) / np.sum(mask) * 100
                        perc_seq.append(percent)
                percents.append(np.mean(perc_seq) if perc_seq else np.nan)
            df = pd.DataFrame({"saliency_threshold": thresholds, "average_percentage": percents})
            csv_name = f"normalized_saliency_threshold_binary_{property_names[j].replace(' ', '_')}.csv"
            df = get_or_save_csv(csv_name, df, input_dir, output_dir)
            
            plt.figure()
            plt.plot(df["saliency_threshold"], df["average_percentage"], marker='o')
            plt.xlabel("Saliency Threshold (theta)")
            plt.ylabel("Average Percentage")
            plt.title(f"Normalized Saliency Threshold (Binary): {property_names[j]}")
            plt.tight_layout()
            png_name = f"normalized_saliency_threshold_binary_{property_names[j].replace(' ', '_')}.png"
            plt.savefig(os.path.join(output_dir, png_name))
            plt.close()
        else:

            p_max = max(np.max(p) for p in prop_list)
            prop_thresholds = np.linspace(0, p_max, 50)
            sal_thresholds = np.linspace(0, 1, 50)
            data_rows = []
            for pt in prop_thresholds:
                for st in sal_thresholds:
                    perc_seq = []
                    for p, sal in zip(prop_list, sal_list):
                        max_sal = np.max(sal) if np.max(sal) > 0 else 1
                        sal_norm = sal / max_sal
                        mask = (p > pt)
                        if np.sum(mask) > 0:
                            percent = np.sum(sal_norm[mask] > st) / np.sum(mask) * 100
                            perc_seq.append(percent)
                    avg_percent = np.mean(perc_seq) if perc_seq else np.nan
                    data_rows.append({"property_threshold": pt, "saliency_threshold": st, "average_percentage": avg_percent})
            df = pd.DataFrame(data_rows)
            csv_name = f"normalized_saliency_threshold_continuous_{property_names[j].replace(' ', '_')}.csv"
            df = get_or_save_csv(csv_name, df, input_dir, output_dir)
            
            pivot_df = df.pivot(index="property_threshold", columns="saliency_threshold", values="average_percentage")
            plt.figure()
            plt.imshow(pivot_df, aspect="auto", origin="lower", 
                       extent=[sal_thresholds[0], sal_thresholds[-1], prop_thresholds[0], prop_thresholds[-1]],
                       cmap="viridis")
            plt.colorbar(label="Average Percentage")
            plt.xlabel("Saliency Threshold")
            plt.ylabel("Property Threshold")
            plt.title(f"Normalized Saliency Threshold (Continuous): {property_names[j]}")
            plt.tight_layout()
            png_name = f"normalized_saliency_threshold_continuous_{property_names[j].replace(' ', '_')}.png"
            plt.savefig(os.path.join(output_dir, png_name))
            plt.close()

def run_all_plots(saliency_maps, properties, property_names, output_dir, input_dir=None):
    """
    Execute all plotting functions.
    """
    plot_scatter_continuous(saliency_maps, properties, property_names, output_dir, input_dir)
    plot_boxplot_binary(saliency_maps, properties, property_names, output_dir, input_dir)
    plot_grouped_bar_chart(saliency_maps, properties, property_names, output_dir, input_dir)
    plot_correlation_heatmap(saliency_maps, properties, property_names, output_dir, input_dir)
    plot_overlay_on_sequence(saliency_maps, properties, property_names, output_dir, input_dir)
    plot_composite_multi_panel(saliency_maps, properties, property_names, output_dir, input_dir)
    plot_pairwise_scatter_matrix(saliency_maps, properties, property_names, output_dir, input_dir)
    plot_normalized_saliency_threshold(saliency_maps, properties, property_names, output_dir, input_dir)

def main():
    parser = argparse.ArgumentParser(
        description="Visualize saliency maps and nucleotide properties."
    )
    parser.add_argument(
        "--saliency_maps",
        type=str,
        required=True,
        help="Path to file containing a list of saliency maps (torch tensors).",
    )
    parser.add_argument(
        "--properties",
        type=str,
        required=True,
        help="Path to file containing a list of property arrays (numpy arrays).",
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

    saliency_maps = torch.load(args.saliency_maps)
    properties = np.load(args.properties, allow_pickle=True)
    property_names = [name.strip() for name in args.property_names.split(",")]

    os.makedirs(args.output_dir, exist_ok=True)
    if args.input_dir is not None:
        os.makedirs(args.input_dir, exist_ok=True)
    run_all_plots(saliency_maps, properties, property_names, args.output_dir, args.input_dir)

if __name__ == "__main__":
    main()

