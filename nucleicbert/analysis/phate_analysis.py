import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import PreTrainedTokenizerFast
import tqdm
from nucleicbert.models.bert import BERT, NB_CONFIG
from nucleicbert.pretrain.pretrainingdataset import RNASeqDataset
import phate
from sklearn.preprocessing import LabelEncoder
import os
import plotly.express as px

import matplotlib

import juelich_colors as jc

def set_style():
    """Set matplotlib style for publication-quality plots."""
    font = {
        'family': 'sans-serif',
        # 'weight': 'bold',
        'size': 25
    }
    axes = {
        'titlesize': 25,
        'labelsize': 25,
        # 'labelweight': 'bold',
        # 'titleweight': 'bold'
    }
    xtick = {
        'labelsize': 25,
    }
    ytick = {
        'labelsize': 25,
    }
    legend = {
        'fontsize': 15,
        'title_fontsize': 15,
        'markerscale': 2,
    }

    matplotlib.rc('font', **font)
    matplotlib.rc('axes', **axes)
    matplotlib.rc('xtick', **xtick)
    matplotlib.rc('ytick', **ytick)
    matplotlib.rc('legend', **legend)

def load_model(
        state_dict_path: str,
        config: dict = NB_CONFIG,
        device: str = 'cuda',
        ) -> BERT:
    """
    Load the model from the state dict path.
    Args:
        state_dict_path: The path to the state dict file.
        config: The config object.
        device: The device to load the model on. Default is 'cuda'.
    Returns:
        model: The loaded model.
    """
    model = BERT(**config)
    if state_dict_path:
        print(f"Loading model from {state_dict_path}")
        model.load_state_dict(torch.load(state_dict_path, map_location=device))
    model.to(device)    
    return model

def load_data(
        data_path: str,
        tokenizer: PreTrainedTokenizerFast,
        length: int = 1024,
        ) -> tuple:
    """
    Load the data from the data path.
    Args:
        data_path: The path to the data file.
        tokenizer: The tokenizer to use for processing sequences.
    Returns:
        dataset: The processed dataset.
        labels: List of labels.
        data: The original dataframe.
    """
    data = pd.read_csv(data_path)
    sequences = data['sequence'].tolist()
    labels = data['type'].tolist()
    dataset = RNASeqDataset(
        input=sequences,
        tokenizer=tokenizer,
        mask_lm_prob=0.15,
        max_length=length,
    )
    return dataset, labels, data

def extract_embds(
        model: BERT,
        dataset: RNASeqDataset,
        device: str = 'cuda',
        length: int = 1024,
        ) -> torch.Tensor:
    """
    Extract the embeddings from the model.
    Args:
        model: The model to extract the embeddings from.
        dataset: The dataset to extract the embeddings from.
        device: The device to load the model on. Default is 'cuda'.
    Returns:
        embeddings: The extracted embeddings.
    """
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
    )
    embeddings_list = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader):
            output = model(batch[0].to(device), output_attentions=True)
            input_list = output[2] # list of [B, 1, L, Hidden_Dim]
            embeddings = torch.cat(input_list, dim=1) # [B, Hidden_Layers+1, L, Hidden_Dim]
            embeddings = embeddings[:, :, :length, :] # [B, Hidden_Layers+1, L, Hidden_Dim]
            embeddings = torch.mean(embeddings, dim=1) # [B, L, Hidden_Dim]
            embeddings = torch.mean(embeddings, dim=1) # [B, Hidden_Dim]
            embeddings_list.append(embeddings)
    embeddings = torch.cat(embeddings_list, dim=0)
    return embeddings

def run_phate_analysis(
        embeddings: torch.Tensor, 
        labels: list,
        output_dir: str = './results',
        n_components: int = 2,
        knn: int = 5,
        decay: int = 40,
        t: int = None,
        n_landmarks: int = 2000,
        random_state: int = 42,
        ) -> np.ndarray:
    """
    Run PHATE analysis on the embeddings and create publication-quality plots.
    Args:
        embeddings: The embeddings to run PHATE on.
        labels: The labels for the embeddings.
        output_dir: The directory to save the results to.
        n_components: Number of dimensions for the embedding.
        knn: Number of nearest neighbors for graph construction.
        decay: Kernel width for the decay rate of kernel similarity.
        t: Diffusion time scale for optimal embedding.
        n_landmarks: Number of landmarks to use for large-scale PHATE.
        random_state: Random seed for reproducibility.
    Returns:
        phate_embeddings: The PHATE embeddings.
    """

    if isinstance(embeddings, torch.Tensor):
        embeddings_np = embeddings.cpu().numpy()
    else:
        embeddings_np = embeddings


    os.makedirs(output_dir, exist_ok=True)
    

    np.save(os.path.join(output_dir, 'original_embeddings.npy'), embeddings_np)
    

    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    unique_labels = le.classes_
    

    n_labels = len(unique_labels)
    palette = jc.set_seaborn_palette()
    

    print("Running PHATE dimensionality reduction...")
    phate_operator = phate.PHATE(
        n_components=n_components,
        knn=knn,
        decay=decay,
        t=t,
        n_landmark=n_landmarks,
        random_state=random_state,
        verbose=True
    )
    
    phate_embeddings = phate_operator.fit_transform(embeddings_np)
    

    np.save(os.path.join(output_dir, 'phate_embeddings.npy'), phate_embeddings)
    

    plt.figure(figsize=(7, 5))
    

    for i, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        plt.scatter(
            phate_embeddings[mask, 0],
            phate_embeddings[mask, 1],
            c=[palette[i]],
            label=label,
            alpha=1.0,
            s=60, 
            # edgecolors='white',
            # linewidth=0.5
        )
    
    plt.legend(
        title="RNA Types", 
        bbox_to_anchor=(1.05, 1), 
        loc='upper left',
        frameon=True,
        fancybox=True,
        shadow=True
    )
    plt.title('Pretrained Model PHATE Projection')
    plt.xlabel('PHATE Dimension 1')
    plt.ylabel('PHATE Dimension 2')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rna_phate_visualization.svg'), 
                dpi=300, transparent=True, format='svg')
    plt.close()  # Close to free memory
    
    print(f"PHATE analysis complete. Results saved to {output_dir}")
    return phate_embeddings

def create_interactive_2d_phate(
        embeddings: np.ndarray,
        data_df: pd.DataFrame,
        labels: list,
        output_dir: str,
        knn: int = 5,
        decay: int = 40,
        t: int = None,
        n_landmarks: int = 2000,
        random_state: int = 42
    ):
    """
    Create an interactive 2D PHATE visualization with hover information.
    
    Args:
        embeddings: Embeddings numpy array
        data_df: DataFrame containing RNA sequence data
        labels: List of labels for the sequences
        output_dir: Directory to save the output files
        knn: Number of nearest neighbors
        decay: Kernel width for decay rate
        t: Diffusion time scale
        n_landmarks: Number of landmarks for large-scale PHATE
        random_state: Random seed
    """

    os.makedirs(output_dir, exist_ok=True)
    

    data_df['sequence_length'] = data_df['sequence'].apply(len)
    

    print("Running 2D PHATE dimensionality reduction for interactive plot...")
    phate_operator = phate.PHATE(
        n_components=2,  # Use 2 dimensions for 2D plot
        knn=knn,
        decay=decay,
        t=t,
        n_landmark=n_landmarks,
        random_state=random_state,
        verbose=True
    )
    

    if isinstance(embeddings, torch.Tensor):
        embeddings_np = embeddings.cpu().numpy()
    else:
        embeddings_np = embeddings
    
    phate_2d = phate_operator.fit_transform(embeddings_np)
    

    np.save(os.path.join(output_dir, 'phate_2d_interactive_embeddings.npy'), phate_2d)
    

    plot_df = pd.DataFrame({
        'PHATE1': phate_2d[:, 0],
        'PHATE2': phate_2d[:, 1],
        'Type': labels,
        'Length': data_df['sequence_length'],
        'Sequence_Preview': data_df['sequence'].apply(lambda s: s[:50] + "..." if len(s) > 50 else s),
        'Index': range(len(labels))
    })
    

    print("Creating interactive 2D plot...")
    
    fig = px.scatter(
        plot_df, 
        x='PHATE1', 
        y='PHATE2',
        color='Type',
        hover_data={
            'Type': True,
            'Length': True,
            'Sequence_Preview': True,
            'Index': True,
            'PHATE1': ':.3f',
            'PHATE2': ':.3f'
        },
        opacity=0.8,
        color_discrete_sequence=px.colors.qualitative.Set2,
        title='Interactive 2D PHATE Projection of RNA Sequences'
    )
    

    fig.update_layout(
        title={
            'text': 'Interactive 2D PHATE Projection of RNA Sequences',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24}
        },
        xaxis_title='PHATE Dimension 1',
        yaxis_title='PHATE Dimension 2',
        width=1200,
        height=800,
        font=dict(size=14),
        legend=dict(
            title="RNA Types",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial",
            bordercolor="gray"
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    

    fig.update_traces(
        marker=dict(
            size=8,
            line=dict(width=0.5, color='white')
        )
    )

    fig.update_traces(
        hovertemplate="<b>RNA Type:</b> %{customdata[0]}<br>" +
                     "<b>Sequence Length:</b> %{customdata[1]}<br>" +
                     "<b>Sequence Preview:</b> %{customdata[2]}<br>" +
                     "<b>Index:</b> %{customdata[3]}<br>" +
                     "<b>PHATE1:</b> %{customdata[4]}<br>" +
                     "<b>PHATE2:</b> %{customdata[5]}<extra></extra>"
    )
    

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    

    html_path = os.path.join(output_dir, 'rna_phate_2d_interactive.html')
    fig.write_html(html_path)
    
    print(f"2D interactive visualization complete. Results saved to: {html_path}")
    return html_path, phate_2d

def run_phate_parameter_sweep(
        embeddings: np.ndarray,
        labels: list,
        output_dir: str,
        knn_list: list = [5, 10, 15],
        decay_list: list = [10, 40, 80],
        t_list: list = [None, 5, 10, 20]
    ):
    """
    Run a parameter sweep for PHATE to find optimal parameters.
    
    Args:
        embeddings: The embeddings to run PHATE on
        labels: The labels for the embeddings
        output_dir: Base directory to save results
        knn_list: List of knn values to try
        decay_list: List of decay values to try
        t_list: List of t values to try
    """
    print("Running PHATE parameter sweep...")
    os.makedirs(output_dir, exist_ok=True)
    

    summary_file = os.path.join(output_dir, 'parameter_sweep_summary.csv')
    summary_df = pd.DataFrame(columns=['knn', 'decay', 't', 'output_dir'])
    
    sweep_count = 0
    for knn in knn_list:
        for decay in decay_list:
            for t in t_list:

                param_dir = os.path.join(output_dir, f"knn_{knn}_decay_{decay}_t_{t if t is not None else 'auto'}")
                os.makedirs(param_dir, exist_ok=True)
                
                print(f"Running configuration: knn={knn}, decay={decay}, t={t}")
                

                run_phate_analysis(
                    embeddings=embeddings,
                    labels=labels,
                    output_dir=param_dir,
                    knn=knn,
                    decay=decay,
                    t=t,
                    random_state=42
                )
                

                summary_df.loc[sweep_count] = [knn, decay, t, param_dir]
                sweep_count += 1
    

    summary_df.to_csv(summary_file, index=False)
    print(f"Parameter sweep complete. Summary saved to {summary_file}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run PHATE analysis on RNA sequences.")
    parser.add_argument('--pretrained', '-p', action='store_true', help='Use this flag if the model is pretrained.')
    parser.add_argument('--length','-l', type=int, default=20, help='Length of RNA sequences to process.')
    parser.add_argument('--general', '-g', action='store_true', help='Use this flag for to run analysis on general RNA sequences of varied lengths.')
    args = parser.parse_args()

    sns.set_context("paper", font_scale=1.5)
    if args.pretrained:
        state_dict_path = "weights/pretrained.pt"
        name = 'pretrained'
    else:
        state_dict_path = None
        name = 'random'
    if args.general:
        length = 1022
        data_path = '../mars_data/processed_data_python/validation_rna_sequences.csv'
    else:
        length = args.length
        data_path = f'../mars_data/processed_data_python/validation_rna_sequences_length_{length}.csv'
    tokenizer_path = 'nucleicbert/tokenizers/noncoding_seqs.json'
    output_dir = f'../mars_data/phate_results_{length}_{name}'
    output_dir_sweep = f'../mars_data/phate_parameter_sweep_{length}_{name}'

    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    

    print("Loading tokenizer...")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    
    print("Loading model...")
    model = load_model(state_dict_path, device=device)
    
    print("Loading data...")
    dataset, labels, data_df = load_data(data_path, tokenizer=tokenizer, length=length+2)
    
    embeddings_path = os.path.join(output_dir, 'original_embeddings.npy')
    if os.path.exists(embeddings_path):
        print("Loading precomputed embeddings...")
        embeddings = np.load(embeddings_path)
    else:
        print("Extracting embeddings...")
        embeddings = extract_embds(model, dataset, device=device, length=length+2)

        os.makedirs(output_dir, exist_ok=True)
        np.save(embeddings_path, embeddings.cpu().numpy())
        embeddings = embeddings.cpu().numpy()
    

    print("Running standard PHATE analysis...")
    phate_result = run_phate_analysis(
        embeddings=embeddings,
        labels=labels,
        output_dir=output_dir,
        knn=10,
        decay=80,
        t=5, 
        n_landmarks=2000 if len(labels) > 2000 else None,
        random_state=42
    )
    

    print("\nCreating interactive 2D visualization...")
    html_path, interactive_embeddings = create_interactive_2d_phate(
        embeddings=embeddings,
        data_df=data_df,
        labels=labels,
        output_dir=output_dir,
        knn=10,
        decay=80,
        t=5,
        n_landmarks=2000 if len(labels) > 2000 else None,
        random_state=42
    )
    
    # Optional: Run parameter sweep to find optimal PHATE parameters
    # Uncomment this section if you want to run the parameter sweep
    
    # print("\nRunning parameter sweep to find optimal PHATE parameters...")
    # run_phate_parameter_sweep(
    #     embeddings=embeddings,
    #     labels=labels,
    #     output_dir=output_dir_sweep,
    #     knn_list=[5, 10, 15],
    #     decay_list=[10, 40, 80],
    #     t_list=[5, 10, 20, 40]
    # )
    
    
    print(f"\nAnalysis complete!")
    print(f"Static plots saved to: {output_dir}")
    print(f"Interactive plot saved to: {html_path}")

if __name__ == '__main__':
    main()