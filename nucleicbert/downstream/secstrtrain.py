import pytorch_lightning as pl
from transformers import PreTrainedTokenizerFast

import nucleicbert.pretrain.train as train
from nucleicbert.downstream.secstrmodule import SecondaryStructure2DPredictor
from nucleicbert.downstream.secstrdataset import RNASecStr2DDataset


if __name__ == '__main__':
    pl.seed_everything(42, workers=True)
    args = train.get_args()
    config = train.get_configs(args.yaml_file)
    

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=config.run_config['tokenizer_file'])
    

    train_dataset = RNASecStr2DDataset(
        data_dir=config.train_data_config['input_dir'],
        tokenizer=tokenizer,
        max_length=config.train_data_config['max_length'],
        min_length=config.train_data_config['min_length']
    )
    
    datasets = [train_dataset]
    

    if config.val_data_config is not None:
        val_dataset = RNASecStr2DDataset(
            data_dir=config.val_data_config['input_dir'],
            tokenizer=tokenizer,
            max_length=config.val_data_config['max_length'],
            min_length=config.val_data_config['min_length']
        )
        datasets.append(val_dataset)
    

    train.run(args, config, SecondaryStructure2DPredictor, datasets, tokenizer)