import pandas as pd
import ast
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os
import argparse


def compute_baselines_and_correlation(ref_csv, target_csv, cutoffs):
    """
    Compute baseline fitness and correlations for given cutoff(s).

    Returns:
        df_target (pd.DataFrame): target DataFrame with added baseline and error columns.
        correlations (list of float): Pearson r for each cutoff.
    """
    df_ref = pd.read_csv(ref_csv)
    df_target = pd.read_csv(target_csv)

    df_ref['mutations'] = df_ref['mutation_position'].apply(ast.literal_eval).apply(set)
    df_target['mutations'] = df_target['mutation_position'].apply(ast.literal_eval).apply(set)

    cutoffs = [cutoffs] if isinstance(cutoffs, int) else list(cutoffs)
    n_target = len(df_target)
    n_ref = len(df_ref)
    distances = np.zeros((n_target, n_ref), dtype=int)
    for i, tset in enumerate(df_target['mutations']):
        distances[i, :] = [len(tset.symmetric_difference(rset)) for rset in df_ref['mutations']]

    correlations = []
    for c in cutoffs:
        baseline_col = f'fit_mean_baseline_{c}'
        error_col = f'abs_error_{c}'
        baselines = []
        for i in range(n_target):
            mask = distances[i] < c
            if mask.any():
                bm = df_ref.loc[mask, 'fit_mean'].mean()
            else:
                bm = np.nan
            baselines.append(bm)
        df_target[baseline_col] = baselines
        df_target[error_col] = np.abs(df_target['fit_mean'] - df_target[baseline_col])

        valid = df_target[[ 'fit_mean', baseline_col ]].dropna()
        if len(valid) < 2:
            correlations.append(np.nan)
        else:
            r, _ = pearsonr(valid['fit_mean'], valid[baseline_col])
            correlations.append(r)

    df_target.drop(columns=['mutations'], inplace=True)
    return df_target, correlations


def visualize_results(updated_csv, cutoffs, output_dir):
    """
    Generate scatter plots of true vs predicted values for each cutoff,
    save plots and CSV data for pgfplots to output_dir.
    """
    df = pd.read_csv(updated_csv)
    cutoffs = [cutoffs] if isinstance(cutoffs, int) else list(cutoffs)
    os.makedirs(output_dir, exist_ok=True)

    for c in cutoffs:
        baseline_col = f'fit_mean_baseline_{c}'
        error_col = f'abs_error_{c}'
        data = df[[ 'fit_mean', baseline_col, error_col ]].dropna()
        x = data['fit_mean']
        y = data[baseline_col]

        plt.figure()
        plt.scatter(x, y)
        maxv = max(x.max(), y.max())
        plt.plot([0, maxv], [0, maxv])
        plt.xlabel('True fit_mean')
        plt.ylabel(f'Predicted fit_mean (cutoff={c})')
        plt.title(f'True vs Predicted (cutoff={c})')
        plot_path = os.path.join(output_dir, f'scatter_cutoff_{c}.png')
        plt.savefig(plot_path)
        plt.close()

        pgf_df = data[[ 'fit_mean', baseline_col ]]
        pgf_csv = os.path.join(output_dir, f'pgf_data_cutoff_{c}.csv')
        pgf_df.to_csv(pgf_csv, index=False)


def main():
    parser = argparse.ArgumentParser(
        description='Compute baseline fitness, errors, correlations, and visualize results.'
    )
    parser.add_argument('--ref', required=True, help='Reference CSV file path')
    parser.add_argument('--target', required=True, help='Target CSV file path')
    parser.add_argument('--cutoff', required=True,
                        help='Single int or comma-separated list of ints for cutoff distances')
    parser.add_argument('--out_csv', required=True, help='Path to save updated target CSV')
    parser.add_argument('--visualize', action='store_true', help='Generate plots and pgf CSVs')
    parser.add_argument('--out_dir', default='results', help='Directory to save plots and pgf CSVs')
    args = parser.parse_args()

    if ',' in args.cutoff:
        cutoffs = [int(x) for x in args.cutoff.split(',')]
    else:
        cutoffs = int(args.cutoff)

    updated_df, correlations = compute_baselines_and_correlation(
        args.ref, args.target, cutoffs)
    updated_df.to_csv(args.out_csv, index=False)
    for c, r in zip((cutoffs if isinstance(cutoffs, list) else [cutoffs]), correlations):
        print(f'Cutoff {c}: Pearson r = {r:.4f}')

    if args.visualize:
        visualize_results(args.out_csv, cutoffs, args.out_dir)
        print(f'Plots and pgf CSVs saved to {args.out_dir}')

if __name__ == '__main__':
    main()
