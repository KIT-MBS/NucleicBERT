import pytorch_lightning as pl
from transformers import PreTrainedTokenizerFast


from nucleicbert.downstream.splicesitemodule import SpliceSitePredictor
from nucleicbert.downstream.splicesitedataset import RNASpliceSiteDataset
import nucleicbert.pretrain.train as train


if __name__ == '__main__':
    pl.seed_everything(42, workers=True)
    args = train.get_args()
    config = train.get_configs(args.yaml_file)
    
    tokenizer = PreTrainedTokenizerFast(tokenizer_file = config.run_config['tokenizer_file'])
    train_dataset = RNASpliceSiteDataset(
            input_dir = config.train_data_config['input_dir'],
            tokenizer = tokenizer,
            max_length = config.train_data_config['max_length'],
            min_length = config.train_data_config['min_length'],
            mode = 'train',
            task = config.run_config['task'],
    )
    datasets = [train_dataset]

    if config.val_data_config is not None:
        val_dataset = RNASpliceSiteDataset(
            input_dir = config.val_data_config['input_dir'],
            tokenizer = tokenizer,
            max_length = config.val_data_config['max_length'],
            min_length = config.val_data_config['min_length'],
            mode = 'val',
            task = config.run_config['task'],
        )
        datasets.append(val_dataset)
    
    train.run(args, config, SpliceSitePredictor, datasets, tokenizer)
    
