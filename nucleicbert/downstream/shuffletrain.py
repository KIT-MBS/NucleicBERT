import pytorch_lightning as pl
from transformers import PreTrainedTokenizerFast


from nucleicbert.downstream.shufflemodule import ShuffleDetectorModule
from nucleicbert.downstream.shuffledataset import RNAShuffleDetectionDataset
import nucleicbert.pretrain.train as train


if __name__ == '__main__':
    pl.seed_everything(42, workers=True)
    args = train.get_args()
    config = train.get_configs(args.yaml_file)
    
    tokenizer = PreTrainedTokenizerFast(tokenizer_file = config.run_config['tokenizer_file'])
    train_dataset = RNAShuffleDetectionDataset(
            input_seq = config.train_data_config['input_dir'],
            tokenizer = tokenizer,
            max_length = config.train_data_config['max_length'],
            min_length = config.train_data_config['min_length'],
            augmentation_factor = config.run_config['augmentation_factor'],
            shuffle_method = config.run_config['shuffle_method'],
    )
    datasets = [train_dataset]
    if config.val_data_config is not None:
        val_dataset = RNAShuffleDetectionDataset(
            input_seq = config.val_data_config['input_dir'],
            tokenizer = tokenizer,
            max_length = config.val_data_config['max_length'],
            min_length = config.val_data_config['min_length'],
            augmentation_factor=1,
            enable_shuffling=False,
        )
        datasets.append(val_dataset)
    
    train.run(args, config, ShuffleDetectorModule, datasets, tokenizer)
    
