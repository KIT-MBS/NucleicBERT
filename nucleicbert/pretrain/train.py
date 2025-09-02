import argparse
import os
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, ConcatDataset
from transformers import PreTrainedTokenizerFast

from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_only


from nucleicbert.pretrain.pretrainingdataset import RNASeqDataset
from nucleicbert.pretrain.pretrainingmodule import PreTrainingModule
import nucleicbert.pretrain.utils as utils



def get_train_dataloader(config, dataset):
    train_dataloader = DataLoader(
        dataset,
        shuffle = True,
        batch_size = config.run_config['batch_size'],
        num_workers = config.run_config['num_workers'],
        persistent_workers = True
    )
    return train_dataloader

def get_val_dataloader(config, dataset):
    val_dataloader = DataLoader(
        dataset,
        shuffle = False,
        batch_size = config.run_config['batch_size'],
        num_workers = config.run_config['num_workers'],
        persistent_workers = True
    )
    return val_dataloader

def get_data_loaders(config, dataset):

    train_size = int(config.run_config['data_split_ratio'] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = get_train_dataloader(config, train_dataset)
    val_dataloader = get_val_dataloader(config, val_dataset)
    torch.save(train_dataset.indices, 'train_dataset_indices.pt')
    torch.save(val_dataset.indices, 'val_dataset_indices.pt')
    return train_dataloader, val_dataloader

def get_loggers(config, resume, save):
    loggers = []
    loggers.append(
        WandbLogger(
            project = config.logging_config['project'],
            name = config.logging_config['name'],
            save_dir = config.logging_config['save_dir'],
            config = config,
            resume = resume,
            log_model = save
        )
    )
    return loggers

def get_trainer(config, loggers, callbacks=None):
    trainer =  pl.Trainer(
        **config.trainer_config,
        max_epochs = config.run_config['max_epochs'],
        logger = loggers,
        callbacks = callbacks
    )
    return trainer


class ModelAndInnerModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _save_checkpoint(self, trainer, filepath):
        super()._save_checkpoint(trainer, filepath)
        
        inner_model_path = filepath.replace('.ckpt', '_inner_model.pt')
        torch.save(trainer.lightning_module.model.state_dict(), inner_model_path)


def get_callbacks(config):
    dirpath = os.path.join(config.run_config['ckpt_dir'], config.logging_config['project'], config.logging_config['name'])
    callbacks = [
        ModelAndInnerModelCheckpoint(
            monitor='Validation Loss',                
            dirpath=dirpath,              
            filename='{epoch:02d}-{Validation Loss:.3f}',  
            save_top_k=1,                      
            mode='min',                        
            save_weights_only=False            
        ),
        ModelAndInnerModelCheckpoint(
            dirpath=dirpath,
            filename='last-{epoch:02d}',
            save_last=True,
            save_weights_only=False
        )
    ]

    return callbacks

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save', type=bool, default=False, help="Whether to save the model or not.")
    parser.add_argument('-lg', '--logging', type=bool, default = False, help="Whether you want to log details from the training process.")
    parser.add_argument('-r', '--resume', type=bool, default=False, help='Whether you want to resume training from a checkpoint.')
    parser.add_argument('-i', '--info', type=str, default=None, help='Additional info for the model name.')
    parser.add_argument('-y', '--yaml_file', type=str, help='YAML file containing various configs', required=True)
    args = parser.parse_args()
    return args

def get_configs(yaml_file):
    config = utils.load_config(yaml_file)
    return config


@rank_zero_only
def save_model(dirpath: str, model: torch.nn.Module):
    utils.save_state_dict(dirpath, model)

def run(args, config, module_type, datasets, tokenizer):
    dirpath = os.path.join(config.run_config['ckpt_dir'], config.logging_config['project'], config.logging_config['name'])
    if len(datasets) == 1:
        dataset = datasets[0]
        train_dataloader, val_dataloader = get_data_loaders(config, dataset)
    if len(datasets) == 2:
        train_dataset, val_dataset = datasets
        train_dataloader = get_train_dataloader(config, train_dataset)
        val_dataloader = get_val_dataloader(config, val_dataset)
    loggers = get_loggers(config, args.resume, args.save)
    callbacks = get_callbacks(config)
    trainer = get_trainer(config, loggers, callbacks)

    if args.resume:
        training_module = module_type.load_from_checkpoint(
            checkpoint_path = utils.CheckpointHandler.find_latest_checkpoint(dirpath, args.resume),
            model_config = config.model_config,
            run_config = config.run_config,
            tokenizer = tokenizer
        )
    else:
        training_module = module_type(config.model_config, config.run_config, tokenizer=tokenizer)

    loggers[0].watch(training_module, log='all')
    try:
        trainer.fit(training_module, train_dataloader, val_dataloader)
        if args.save:
            print('Saving the model.')
            if module_type == PreTrainingModule:
                save_model(dirpath, training_module.model)
            else:
                save_model(dirpath, training_module)
            # utils.save_state_dict(dirpath, training_module.model)
    except Exception as e:
        print('Training failed.')
        if args.save:
            print('Saving the model.')
            if module_type == PreTrainingModule:
                save_model(dirpath, training_module.model)
            else:
                save_model(dirpath, training_module)
        raise e

if __name__ == '__main__':
    pl.seed_everything(42, workers=True)
    args = get_args()
    config = get_configs(args.yaml_file)
    
    tokenizer = PreTrainedTokenizerFast(tokenizer_file = config.run_config['tokenizer_file'])
    train_dataset = RNASeqDataset(
        input = config.train_data_config['input_dir'],
        tokenizer = tokenizer,
        constant_mask_positions = config.run_config['constant_mask_positions'],
        max_length = config.model_config['max_length'],
        mask_lm_prob = config.run_config['mask_lm_prob'],
        use_span_masking=False,
        span_masking_ratio=0.5,
        max_span_length=8,
        min_span_length=4,
    )
    datasets = [train_dataset]
    if config.val_data_config is not None:
        val_dataset = RNASeqDataset(
            input = config.val_data_config['input_dir'],
            tokenizer = tokenizer,
            constant_mask_positions = config.run_config['constant_mask_positions'],
            max_length = config.model_config['max_length'],
            mask_lm_prob = config.run_config['mask_lm_prob'],
            use_span_masking=False,
            span_masking_ratio=0.5,
            max_span_length=8,
            min_span_length=4,
        )
        datasets.append(val_dataset)
    
    run(args, config, PreTrainingModule, datasets, tokenizer)
    
