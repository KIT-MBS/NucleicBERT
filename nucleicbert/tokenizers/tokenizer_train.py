from tokenizers.pre_tokenizers import Split
import tokenizers
from tokenizers.normalizers import Lowercase
from tokenizers.trainers import BpeTrainer
from tokenizers.models import BPE
from tokenizers import Tokenizer
from nucleicbert.tokenizers.special_tokens import special_tokens, special_tokens_dict
"""
Use this script to train a tokenizer on a dataset of sequences.
The tokenizer will be saved as a json file.

It requires the following arguments:

--data-files: the path to the data file or directory containing the data files.
    If data-files is a directory, all files in the directory will be used.
--tokenizer-path: the path to save the tokenizer to.
--vocab-size: the size of the vocabulary to train the tokenizer with.
--min-frequency: the minimum frequency of a token in the dataset to be included in the vocabulary.
--clump-size: the number of tokens to clump together when training the tokenizer.


"""




if __name__ == '__main__':
    import argparse
    import glob
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-files', '-d', type=str, required=True)
    parser.add_argument('--tokenizer-path','-p', type=str, required=True)
    parser.add_argument('--vocab-size','-s', type=int, default=1000)
    parser.add_argument('--min-frequency','-f', type=int, default=2)
    parser.add_argument('--clump-size','-c', type=int, default=1)
    args = parser.parse_args()

    # you can use a pattern, file or a dir as input
    files = glob.glob(args.data_files)
    if not files:
        files = [args.data_files]
    if os.path.isdir(args.data_files):
        files = glob.glob(os.path.join(args.data_files, '*'))
    if not files:
        raise FileNotFoundError(f'No files found at {args.data_files}')\
        
    
    clump_size = args.clump_size

    pattern = r'[a-zA-Z]{{{}}}'.format(clump_size)
    normalizer = Lowercase()
    r = tokenizers.Regex(pattern)
    pre_tokenizer = Split(pattern=r, behavior='isolated', invert=False)
    tokenizer = Tokenizer(BPE())
    tokenizer.normalizer = normalizer
    tokenizer.pre_tokenizer = pre_tokenizer
    trainer = BpeTrainer(special_tokens=special_tokens, vocab_size=args.vocab_size, min_frequency=args.min_frequency, show_progress=True)
    
    tokenizer.train(files, trainer=trainer)
    
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.save(args.tokenizer_path)