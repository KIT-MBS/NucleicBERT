import os
import argparse
import torch
import numpy as np
from transformers import PreTrainedTokenizerFast
import pytorch_lightning as pl

from nucleicbert.inference.saliency.generator import SaliencyMapGenerator
from nucleicbert.pretrain.pretrainingdataset import RNASeqDataset
from nucleicbert.downstream.secstrdataset import RNASecStr2DDataset
from nucleicbert.downstream.closenessdataset import RNAClosenessDataset

import sys

def main():
    parser = argparse.ArgumentParser(description='Generate and analyze saliency maps for RNA sequences')
    parser.add_argument('--model', required=True, help='Path to model checkpoint')
    parser.add_argument('--model-type', choices=['bert', 'bert_with_secstr', 'bert_with_closeness'], 
                        default='bert', help='Type of model to analyze')
    parser.add_argument('--data-path', required=True, help='Path to dataset')
    parser.add_argument('--output-dir', default='saliency_outputs', help='Directory to save outputs')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--max-samples', type=int, default=10, help='Maximum number of samples to process')
    parser.add_argument('--tokenizer', default='nucleicbert/tokenizers/noncoding_seqs.json', 
                        help='Path to tokenizer')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to run analysis on')
    parser.add_argument('--skip-individual', action='store_true', 
                        help='Skip individual sequence visualizations (only run aggregate analysis)')
    parser.add_argument('--property-names', default='contact_degree,long_range_ratio', 
                        help='Comma-separated list of property names for extended analysis')
    parser.add_argument('--input-dir', default=None, 
                        help='(Optional) Directory from which to read precomputed CSV files for nucleobert_saliency_analysis')
    parser.add_argument('--secstr-analysis-only', action='store_true',
                        help='Only run secondary structure analysis on pre-computed data')
    args = parser.parse_args()
    
    pl.seed_everything(42, workers=True)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.secstr_analysis_only:
        run_precomputed_secstr_analysis(args)
        return
    
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer)
    
    if args.model_type == 'bert':
        dataset = RNASeqDataset(
            input=args.data_path,
            tokenizer=tokenizer,
            constant_mask_positions=None,
            max_length=1024,
            mask_lm_prob=0.15,
        )
    elif args.model_type == 'bert_with_secstr':
        dataset = RNASecStr2DDataset(
            data_dir=args.data_path,
            tokenizer=tokenizer,
            max_length=1024,
            min_length=20,
        )
    elif args.model_type == 'bert_with_closeness':
        dataset = RNAClosenessDataset(
            input_seq=os.path.join(args.data_path, 'inputs'),
            target=os.path.join(args.data_path, 'targets'),
            tokenizer=tokenizer,
        )
    
    if args.max_samples < len(dataset):
        dataset = torch.utils.data.Subset(dataset, range(args.max_samples))
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    
    generator = SaliencyMapGenerator(
        model_path=args.model,
        model_type=args.model_type,
        device=args.device,
        tokenizer=tokenizer,
    )
    
    print(f"Generating saliency maps for {min(args.max_samples, len(dataset))} samples...")
    saliency_maps = generator.generate_salient_maps(
        data_loader, 
        args.output_dir, 
        args.max_samples,
        individual_visualizations=not args.skip_individual
    )
    
    if args.model_type == 'bert_with_closeness':
        try:
            from nucleobert_saliency_analysis import run_all_plots
            
            print("Running extended saliency analysis...")
            
            saved_saliency_maps = torch.load(os.path.join(args.output_dir, 'all_saliency_maps.pt'))
            saved_contact_maps = torch.load(os.path.join(args.output_dir, 'all_contact_maps.pt'))
            
            property_names = [name.strip() for name in args.property_names.split(",")]
            all_properties = []
            
            for contact_map in saved_contact_maps:
                if hasattr(contact_map, 'detach'):
                    contact = contact_map.squeeze().detach().cpu().numpy()
                else:
                    contact = np.array(contact_map).squeeze()
                
                seq_len = contact.shape[0] # Should be square matrix (NxN)
                

                properties = np.zeros((len(property_names), seq_len))
                
                for i, prop_name in enumerate(property_names):
                    if prop_name == 'contact_degree':
                        properties[i] = np.sum(contact, axis=1)
                    
                    elif prop_name == 'long_range_ratio':
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
                        
                        has_contacts = (short_range_contacts + long_range_contacts) > 0
                        long_range_ratio = np.zeros(seq_len)
                        long_range_ratio[has_contacts] = long_range_contacts[has_contacts] / (short_range_contacts[has_contacts] + long_range_contacts[has_contacts])
                        properties[i] = long_range_ratio
                
                all_properties.append(properties)
            
            extended_output_dir = os.path.join(args.output_dir, 'extended_analysis')
            os.makedirs(extended_output_dir, exist_ok=True)
            
            run_all_plots(
                saved_saliency_maps, 
                all_properties, 
                property_names, 
                extended_output_dir,
                args.input_dir
            )
            
            print(f"Extended analysis complete. Results saved to {extended_output_dir}")
            
        except ImportError:
            print("Could not import nucleobert_saliency_analysis. Skipping extended analysis.")
    
    print(f"Analysis complete. Results saved to {args.output_dir}")

def run_precomputed_secstr_analysis(args):
    """
    Run secondary structure analysis on pre-computed data
    
    Args:
        args: Command line arguments
    """
    try:
        # Check if the saved files exist
        saliency_path = os.path.join(args.output_dir, 'all_saliency_maps.pt')
        secstr_path = os.path.join(args.output_dir, 'all_sequences_structures.npy')
        
        if not os.path.exists(saliency_path) or not os.path.exists(secstr_path):
            print(f"Error: Pre-computed data files not found at {args.output_dir}")
            print("Please run the full analysis first to generate the required data.")
            return
        
        print(f"Loading pre-computed data from {args.output_dir}...")
        
        all_saliency_maps = torch.load(saliency_path)
        
        secstr_data = np.load(secstr_path, allow_pickle=True).item()
        all_sequences = secstr_data['sequences']
        all_structures = secstr_data['structures']
        
        print(f"Loaded {len(all_saliency_maps)} saliency maps with {len(all_sequences)} sequences and structures")
        
        aggregate_dir = os.path.join(args.output_dir, 'aggregate')
        os.makedirs(aggregate_dir, exist_ok=True)
        
        from nucleicbert.inference.saliency.visualizers.secstr_aggregate import run_all_secstr_analyses, compare_saliency_across_rna_properties
        
        print("Running secondary structure analysis...")
        secstr_results = run_all_secstr_analyses(
            all_saliency_maps,
            all_sequences,
            all_structures,
            aggregate_dir,
            prefix='secstr'
        )
        
        print("Running RNA property analysis...")
        rna_prop_results = compare_saliency_across_rna_properties(
            all_saliency_maps,
            all_sequences,
            all_structures,
            aggregate_dir,
            prefix='rna_properties'
        )
        
        print(f"Secondary structure analysis complete. Results saved to {aggregate_dir}")
        
    except ImportError as e:
        print(f"Error importing secondary structure analysis modules: {e}")
        print("Please ensure the nucleicbert.inference.saliency.visualizers.secstr_aggregate module is installed.")
        return

if __name__ == '__main__':
    main()