import pytorch_lightning as pl
from transformers import PreTrainedTokenizerFast

from nucleicbert.downstream.closenessmodule import ClosenessPredictor
from nucleicbert.downstream.closenessdataset import RNAClosenessDataset
import nucleicbert.pretrain.train as train

if __name__ == '__main__':
    pl.seed_everything(42, workers=True)
    args = train.get_args()
    config = train.get_configs(args.yaml_file)
    loggers = train.get_loggers(config, args.resume, args.save)
    callbacks = train.get_callbacks(config)
    trainer = train.get_trainer(config, loggers, callbacks)
    
    tokenizer = PreTrainedTokenizerFast(tokenizer_file = config.run_config['tokenizer_file'])
    train_dataset = RNAClosenessDataset(
            input_seq = config.train_data_config['input_dir'],
            target = config.train_data_config['target_dir'],
            tokenizer = tokenizer,
            remove_backbone_width=config.run_config['remove_backbone_width'],
            distance_threshold=config.run_config['distance_threshold'],
            task=config.run_config['task'],
            max_length = config.train_data_config['max_length'],
            min_length = config.train_data_config['min_length']
    )
    datasets = [train_dataset]
    if config.val_data_config is not None:
        val_dataset = RNAClosenessDataset(
            input_seq = config.val_data_config['input_dir'],
            target = config.val_data_config['target_dir'],
            tokenizer = tokenizer,
            remove_backbone_width=config.run_config['remove_backbone_width'],
            distance_threshold=config.run_config['distance_threshold'],
            task=config.run_config['task'],
            max_length = config.val_data_config['max_length'],
            min_length = config.val_data_config['min_length']
        )
        datasets.append(val_dataset)
    
    train.run(args, config, ClosenessPredictor, datasets, tokenizer)