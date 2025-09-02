import torch
from torch.utils.data import DataLoader, Dataset
import typing
from nucleicbert.pretrain.utils import load_input_lines, truncate, pad


class RNASeqDataset(Dataset):
    """
    This is related to dynamic masking used by hugging face mlm.
    Supports both single-token masking and span masking.
    """
    def __init__(
            self,
            input: list[str],
            tokenizer,
            mask_lm_prob: float = 0.15,
            constant_mask_positions: typing.Optional[typing.List[int]] = None,
            max_length: int = 512,
            min_length: int = 1,
            use_span_masking: bool = False,
            span_masking_ratio: float = 0.5,  # Ratio of samples to use span masking
            max_span_length: int = 3,  # Maximum span length
            min_span_length: int = 1,  # Minimum span length
        ) -> None:
        super(RNASeqDataset, self).__init__()
        self.constant_mask_positions = constant_mask_positions
        self.input_lines = input if isinstance(input, list) else load_input_lines(input)
        self.max_length = max_length
        self.min_length = min_length
        self.tokenizer = tokenizer
        self.mask_lm_prob = mask_lm_prob
        self.use_span_masking = use_span_masking
        self.span_masking_ratio = span_masking_ratio
        self.max_span_length = max_span_length
        self.min_span_length = min_span_length


    def preprocess(self, sequence: str, idx: int = 0)-> typing.Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Preprocesses the sequence for pretrain

        Args:
            sequence: The sequence to preprocess
            idx: The index of the sequence in the dataset
        Returns:
            input_ids: The input ids
            masked_positions: The masked positions
            masked_lm_target_ids: The masked target_ids

        """
        input_ids, target_ids, special_tokens_mask = self.pretokenize(sequence)
        if self.constant_mask_positions:
            input_ids, target_ids = self.apply_constant_masking(input_ids, target_ids, special_tokens_mask)
        else:
            # Determine whether to use span masking or single-token masking based on sequence index
            if self.use_span_masking and (idx % 2 == 0 or torch.rand(1).item() < self.span_masking_ratio):
                input_ids, target_ids = self.apply_span_masking(input_ids, target_ids, special_tokens_mask)
            else:
                input_ids, target_ids = self.apply_dynamic_masking(input_ids, target_ids, special_tokens_mask)
        return input_ids, target_ids, sequence
    
    def apply_constant_masking(self, input_ids: torch.Tensor, target_ids: torch.Tensor, special_tokens_mask)-> typing.Tuple[torch.Tensor, torch.Tensor]:
        """This masks the given positions in the input_ids
        
        NOTE: if one of the constant_mask_position conincides with the special token, it will be ignored
        Args:
            input_ids: The input ids
            target_ids: The target ids
            special_tokens_mask: The special tokens mask

        Returns:
            input_ids: The input ids with the masked positions replaced with [MASK]
            target_ids: The target ids with the masked positions replaced with -100

        Raises: AssertionError if the constant_mask_positions are out of bounds
        """

        constant_mask = torch.zeros(target_ids.shape, dtype=torch.bool)
        constant_mask_positions = torch.tensor(self.constant_mask_positions)
        assert torch.all(constant_mask_positions >= 0) and torch.all(constant_mask_positions < len(constant_mask)), "Out-of-bounds indices in constant_mask_positions"
        constant_mask[constant_mask_positions] = True


        probability_matrix = torch.full(target_ids.shape, 0.0)
        probability_matrix.masked_fill_(constant_mask, value=1.0)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        target_ids[~masked_indices] = -100  # We only compute loss on masked tokens
        indices_replaced = masked_indices
        input_ids[indices_replaced] = self.tokenizer.vocab['[MASK]']

        return input_ids, target_ids
    
    def apply_dynamic_masking(self, input_ids: torch.Tensor, target_ids: torch.Tensor, special_tokens_mask: torch.Tensor)-> typing.Tuple[torch.Tensor, torch.Tensor]:
        #masking process
        probability_matrix = torch.full(target_ids.shape, self.mask_lm_prob)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        target_ids[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(target_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.vocab['[MASK]']

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(target_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(5, len(self.tokenizer.vocab), target_ids.shape, dtype=torch.long) # -5 to avoid special tokens
        input_ids[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return input_ids, target_ids

    def apply_span_masking(self, input_ids: torch.Tensor, target_ids: torch.Tensor, special_tokens_mask: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply span masking following SpanBERT approach: treat all tokens in a span the same way.
        
        Args:
            input_ids: The input ids
            target_ids: The target ids
            special_tokens_mask: The special tokens mask
            
        Returns:
            input_ids: The input ids with masked spans
            target_ids: The target ids with corresponding positions marked for loss computation
        """
        masked_indices = torch.zeros_like(input_ids, dtype=torch.bool)
        
        eligible_positions = (~special_tokens_mask).nonzero(as_tuple=True)[0]
        
        if len(eligible_positions) == 0:
            target_ids[:] = -100
            return input_ids, target_ids
        
        num_tokens_to_mask = int(len(eligible_positions) * self.mask_lm_prob)
        
        num_tokens_to_mask = max(1, num_tokens_to_mask)
        

        used_positions = set()
        
        spans = []
        
        tokens_masked = 0
        max_attempts = 100  # Prevent infinite loops
        attempt = 0
        
        while tokens_masked < num_tokens_to_mask and attempt < max_attempts:
            attempt += 1
            
            perm = torch.randperm(len(eligible_positions))
            pos_idx = eligible_positions[perm[0]].item()
            
            if pos_idx in used_positions:
                continue
                
            span_length = torch.randint(
                self.min_span_length, 
                min(self.max_span_length + 1, len(eligible_positions) - tokens_masked + 1), 
                (1,)
            ).item()
            
            valid_span = True
            span_positions = []
            
            for offset in range(span_length):
                curr_pos = pos_idx + offset
                
                if curr_pos >= len(input_ids) or special_tokens_mask[curr_pos] or curr_pos in used_positions:
                    valid_span = False
                    break
                    
                span_positions.append(curr_pos)
            
            if valid_span and len(span_positions) > 0:
                spans.append(span_positions)
                
                for pos in span_positions:
                    masked_indices[pos] = True
                    used_positions.add(pos)
                    tokens_masked += 1
                    
                    if tokens_masked >= num_tokens_to_mask:
                        break
        
        target_ids = input_ids.clone()
        target_ids[~masked_indices] = -100
        
        for span in spans:
            span_fate = torch.rand(1).item()
            
            if span_fate < 0.8:  # 80% replace with [MASK]
                for pos in span:
                    input_ids[pos] = self.tokenizer.vocab['[MASK]']
            elif span_fate < 0.9:  # 10% replace with random tokens (same random token for all positions would be weird)
                random_tokens = torch.randint(5, len(self.tokenizer.vocab), (len(span),), dtype=torch.long)
                for i, pos in enumerate(span):
                    input_ids[pos] = random_tokens[i]
            # 10% keep unchanged
    
        return input_ids, target_ids

    def pretokenize(self, sequence: str):
        input_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sequence))
        input_ids = truncate(input_ids, self.max_length-2) # -2 for [CLS] and [SEP] othewise the length of sequences will be greater than max_length
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_ids = torch.cat([torch.tensor([self.tokenizer.vocab['[CLS]']]), input_ids, torch.tensor([self.tokenizer.vocab['[SEP]']])], dim=0)
        input_ids = pad(input_ids, self.max_length)
        target_ids = input_ids.clone()

        special_tokens_mask = self.find_special_tokens_mask(
            target_ids.tolist(),[
            self.tokenizer.vocab['[CLS]'],
            self.tokenizer.vocab['[SEP]'],
            self.tokenizer.vocab['[PAD]'],
            self.tokenizer.vocab['[UNK]'],
            ]
        )
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        return input_ids, target_ids, special_tokens_mask

    @staticmethod
    def find_special_tokens_mask(input_ids, special_tokens):
        return [token in special_tokens for token in input_ids]

    def __len__(self)-> int:
        return len(self.input_lines)

    def __getitem__(self, idx)-> typing.Tuple[torch.Tensor, torch.Tensor, str]:
        line = self.input_lines[idx]
        data = self.preprocess(line, idx)
        return data
            
if __name__ == '__main__':
    from transformers import PreTrainedTokenizerFast
    import pytorch_lightning as pl
    pl.seed_everything(42, workers=True)

    input = ['agcugaugcuagcuagcuagcaugaucgagcaugacuagagaagagagagaggaggagaggagagaggggggguugaugaugauggau', 'GAUCGCUGAUGCUACGCUGCAUCGACUG']
    tokenizer = PreTrainedTokenizerFast(tokenizer_file = 'nucleicbert/tokenizers/noncoding_seqs.json')
    dataset = RNASeqDataset(
        input = input,
        tokenizer = tokenizer,
        max_length = 512,
        min_length = 1,
        mask_lm_prob = 0.15,
        use_span_masking = True,
        span_masking_ratio = 0.5,
        max_span_length = 8,
        min_span_length = 4
    )

    dataloader = DataLoader(dataset, shuffle = True, batch_size = 2)
    contact_map_dataloader = DataLoader(dataset, shuffle = False, batch_size = 1)
    for epoch in range(1):
        for batch in contact_map_dataloader:
            a,b,c= batch
            print('Input ids:', a, 'Target ids:', b, c)
            # break