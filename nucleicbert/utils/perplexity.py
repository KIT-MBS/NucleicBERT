import torch
import torch.nn.functional as F
from nucleicbert.pretrainingmodule import PreTrainingModule
from transformers import PreTrainedTokenizerFast
import math
from typing import List, Tuple
from nucleicbert.dataset import RNASeqDataset
import tqdm
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl


def random_masking_perplexity(
    model,
    tokenizer,
    dataloader
) -> float:
    """
    Computes the stochastic random-masking-based perplexity.

    Args:
        model: Pretrained masked language model (e.g., BERT).
        tokenizer: Tokenizer corresponding to the model.
        text: Input text to compute perplexity for.
        mask_prob: Probability of replacing a token with [MASK].
        max_length: Maximum length of the input sequence.
        min_length: Minimum length of the input sequence.

    Returns:
        Estimated perplexity of the sequence.
    """

    individual_perplexities = []

    loss_fn = nn.CrossEntropyLoss(ignore_index = -100, reduction = 'mean')
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader):
            masked_input_ids, target_ids = batch
            masked_input_ids = masked_input_ids.to(model.device)
            target_ids = target_ids.to(model.device)


            # Forward pass
            logits = model(masked_input_ids)  # No attention mask needed
            mask = (target_ids != -100)
            if mask.sum() == 0:
                continue
            masked_logits = logits[mask]
            masked_targets = target_ids[mask]
            loss = loss_fn(masked_logits, masked_targets)
            single_perplexity = torch.exp(loss)
            individual_perplexities.append(single_perplexity.item())
    perplexity_tensor = torch.tensor(individual_perplexities)
    perplexity = torch.mean(perplexity_tensor).item()
    return perplexity


def calculate_pseudoperplexity_single_sequence(model, tokenizer, input_ids):
    """
    Calculate pseudoperplexity for a single input sequence.

    Args:
        model: Pre-trained model (e.g., transformers model).
        tokenizer: Corresponding tokenizer.
        input_ids: Tensor of token IDs for a single sequence.

    Returns:
        Pseudoperplexity (float).
    """
    device = model.device
    model.eval()

    total_log_likelihood = 0.0
    total_tokens = 0

    input_ids = input_ids.to(device)
    
    # Remove [CLS] and [SEP] if present
    cls_token_id = tokenizer.vocab.get('[CLS]')
    sep_token_id = tokenizer.vocab.get('[SEP]')
    pad_token_id = tokenizer.vocab.get('[PAD]')
    valid_token_mask = (input_ids != pad_token_id)
    if cls_token_id and sep_token_id:
        valid_token_mask &= (input_ids != cls_token_id) & (input_ids != sep_token_id)

    # Filter valid tokens
    valid_input_ids = input_ids[valid_token_mask]

    # Sequence length
    seq_len = valid_input_ids.size(0)

    with torch.no_grad():
        for token_idx in range(seq_len):
            # Create a copy of the sequence and mask the current token
            masked_input_ids = valid_input_ids.clone()
            masked_input_ids[token_idx] = tokenizer.vocab['[MASK]']  # Mask current token

            # Forward pass
            logits = model(masked_input_ids.unsqueeze(0))  # Add batch dimension

            # Extract logits for the masked token
            token_logits = logits[0, token_idx, :]  # (vocab_size)
            target_token = valid_input_ids[token_idx]  # Target token ID

            # Compute log likelihood for the target token
            log_likelihood = -F.cross_entropy(
                token_logits.unsqueeze(0), target_token.unsqueeze(0)
            )
            total_log_likelihood += log_likelihood.item()
            total_tokens += 1

    # Calculate pseudoperplexity
    mean_log_likelihood = total_log_likelihood / total_tokens
    perplexity = math.exp(-mean_log_likelihood)

    return perplexity

# Test the implementation
def test_perplexity_methods():
    """Test both perplexity methods on a sample sequence."""
    model_path = 'weights/pretrained.pt'

    model_config = {
        'dropout': 0.1,
        'hidden_size': 1024,
        'max_length': 1024,
        'num_attention_heads': 32,
        'num_hidden_layers': 32,
        'vocab_size': 25,
        'position_embedding': 'learned'
    }
    run_config = {
        'batch_size': 1,
        'lr': 1.0e-03,
        'num_workers': 64,
        'max_epochs': 1000,
        'use_scheduler': False,
        'constant_mask_positions': None,
        'mask_lm_prob': 0.15,
        'weight_decay': 0.0001,
        'tokenizer_file': 'nucleicbert/tokenizers/noncoding_seqs.json'
    }
    tokenizer = PreTrainedTokenizerFast(tokenizer_file = run_config['tokenizer_file'])
    model = PreTrainingModule.load_from_checkpoint(
        model_path,
        model_config=model_config,
        run_config=run_config,
        tokenizer=tokenizer
    )

    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    text = '../data/benchmarking_data/1000/'



    dataset = RNASeqDataset(
        input = text,
        tokenizer = tokenizer,
        mask_lm_prob = run_config['mask_lm_prob'],
        max_length = model_config['max_length'],
        min_length = 1
    )
    input_ids = dataset[0][0]
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size = run_config['batch_size'],
        shuffle = False,
        num_workers = run_config['num_workers']
    )
    print("Calculating random masking perplexity...")
    random_perplexity = random_masking_perplexity(model, tokenizer, dataloader)
    print(f"Random Masking Perplexity: {random_perplexity:.4f}")

    print("Calculating pseudoperplexity for single sequence...")
    pseudoperplex_single = calculate_pseudoperplexity_single_sequence(model, tokenizer, input_ids)
    print(f"Pseudoperplexity for Single Sequence: {pseudoperplex_single:.4f}")


if __name__ == "__main__":
    pl.seed_everything(42)
    # print('Yes')
    test_perplexity_methods()
