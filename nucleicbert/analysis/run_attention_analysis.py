import os
import argparse
import torch
import numpy as np
import pandas as pd
import glob
import pytorch_lightning as pl
from transformers import PreTrainedTokenizerFast
from nucleicbert.analysis.attention.analyzer import AttentionMapAnalyzer
from nucleicbert.models.bert import BERT, NB_CONFIG
from nucleicbert.downstream.closenessmodule import BERTWithClosenessResNet
from nucleicbert.downstream.secstrmodule import SecondaryStructure2DPredictor, BERTWithSecStr2D
from nucleicbert.downstream.shufflemodule import ShuffleDetectorModule

from nucleicbert.downstream.closenessdataset import RNAClosenessDataset
from nucleicbert.downstream.secstrdataset import RNASecStr2DDataset
from nucleicbert.pretrain.pretrainingdataset import RNASeqDataset
from nucleicbert.downstream.shuffledataset import RNAShuffleDetectionDataset

import sys

def main():
    parser = argparse.ArgumentParser(
        description="Analyze attention maps to understand how different layers and heads contribute to predictions."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved model checkpoint.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["bert", "bert_with_secstr", "bert_with_closeness", "bert_with_contactmap", "bert_with_shuffle"],
        required=True,
        help="Type of model architecture."
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path to the tokenizer file."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the data file or directory."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save analysis results."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for data loading."
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=10,
        help="Maximum number of samples to analyze."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Optional directory from which to read precomputed CSV files."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on (cuda or cpu)."
    )
    parser.add_argument(
        "--custom_properties",
        type=str,
        default=None,
        help="Optional path to custom property maps (.npy file)."
    )
    parser.add_argument(
        "--custom_property_names",
        type=str,
        default=None,
        help="Optional comma-separated list of custom property names."
    )
    parser.add_argument(
        "--motifs_csv",
        type=str,
        default=None,
        help="Path to CSV file containing RNA motifs."
    )
    
    args = parser.parse_args()
    

    os.makedirs(args.output_dir, exist_ok=True)
    
 
    if args.model_type == 'bert':
        model = BERT(**NB_CONFIG)
        if args.model_path is not None:
            model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    elif args.model_type == 'bert_with_secstr':

        model_config = {
            'vocab_size': NB_CONFIG['vocab_size'],
            'hidden_size': NB_CONFIG['hidden_size'],
            'num_hidden_layers': NB_CONFIG['num_hidden_layers'],
            'num_attention_heads': NB_CONFIG['num_attention_heads'],
            'dropout': NB_CONFIG['dropout'],
            'max_length': NB_CONFIG['max_length'],
            'position_embedding': NB_CONFIG['position_embedding'],
        }
        
        run_config = {
            'lr': 5e-5,
            'weight_decay': 0.01,
            'max_epochs': 10,
            'internal_logs_freq': 100,
            'pretrained_path': None,  
            'frozen': False,  
            'use_scheduler': True,
        }
        

        tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer)
        

        model = SecondaryStructure2DPredictor(model_config, run_config, tokenizer)
        

        if args.model_path is not None:
            checkpoint = torch.load(args.model_path, map_location=args.device)
            model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
    elif args.model_type == 'bert_with_closeness' or args.model_type == 'bert_with_contactmap':
        bert = BERT(**NB_CONFIG)
        model = BERTWithClosenessResNet(
            bert,
            input_channels=NB_CONFIG['num_attention_heads']*NB_CONFIG['num_hidden_layers'],
            num_residual_blocks=1,
            task='contact_map'
        )
        if args.model_path is not None:
            model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    elif args.model_type == 'bert_with_shuffle':

        model_config = {
            'vocab_size': NB_CONFIG['vocab_size'],
            'hidden_size': NB_CONFIG['hidden_size'],
            'num_hidden_layers': NB_CONFIG['num_hidden_layers'],
            'num_attention_heads': NB_CONFIG['num_attention_heads'],
            'dropout': NB_CONFIG['dropout'],
            'max_length': NB_CONFIG['max_length'],
            'position_embedding': NB_CONFIG['position_embedding'],
        }
        
        run_config = {
            'lr': 5e-5,
            'weight_decay': 0.01,
            'max_epochs': 10,
            'internal_logs_freq': 100,
            'pretrained_path': None,  
            'frozen': False,  
            'use_scheduler': True,
        }
        

        tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer)
        

        model = ShuffleDetectorModule(model_config, run_config, tokenizer)
        

        if args.model_path is not None:
            checkpoint = torch.load(args.model_path, map_location=args.device)
            model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    model = model.to(args.device)
    model.eval()
    

    pl.seed_everything(42, workers=True)
    

    os.makedirs(args.output_dir, exist_ok=True)
    

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
    elif args.model_type == 'bert_with_closeness' or args.model_type == 'bert_with_contactmap':

        dataset = RNAClosenessDataset(
            input_seq=os.path.join(args.data_path, 'inputs'),
            target=os.path.join(args.data_path, 'targets'),
            tokenizer=tokenizer,
        )
    elif args.model_type == 'bert_with_shuffle':

        dataset = RNAShuffleDetectionDataset(
            input_seq=args.data_path,
            tokenizer=tokenizer,
            max_length=1024,
            min_length=20,
        )
    

    if args.max_samples < len(dataset):
        dataset = torch.utils.data.Subset(dataset, range(args.max_samples))
    

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    

    analyzer = AttentionMapAnalyzer(
        model,
        args.model_type,
        device=args.device,
        tokenizer=tokenizer,
        motifs_csv_path=args.motifs_csv,  
    )
    

    custom_properties = None
    custom_property_names = None
    
    if args.custom_properties and args.custom_property_names:
        custom_property_names = [name.strip() for name in args.custom_property_names.split(",")]
        custom_properties = {}
        

        for prop_name in custom_property_names:
            prop_dir = os.path.join(args.custom_properties, "properties", prop_name)
            if not os.path.exists(prop_dir):
                prop_dir = os.path.join(args.custom_properties, prop_name)
                if not os.path.exists(prop_dir):
                    print(f"Warning: Property directory not found for {prop_name}")
                    continue
            

            sample_files = sorted(glob.glob(os.path.join(prop_dir, "sample_*.csv")))
            if not sample_files:
                print(f"Warning: No sample CSV files found for property {prop_name}")
                continue
            

            prop_list = []
            for sample_file in sample_files:
                df = pd.read_csv(sample_file)
                

                if len(df.columns) == 1:  # 1D property
                    prop_list.append(df['value'].values)
                else:  # 2D property
                    prop_list.append(df.values)
            
            custom_properties[prop_name] = prop_list
    

    analyzer.analyze_attention_for_data(
        data_loader,
        args.output_dir,
        max_samples=args.max_samples,
        input_dir=args.input_dir,
        custom_properties=custom_properties,
        custom_property_names=custom_property_names,
    )
    
    print(f"Analysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()