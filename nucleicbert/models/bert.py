import torch
import torch.nn as nn
import math
import typing


NB_CONFIG = {
    'vocab_size': 25,
    'hidden_size': 1024,
    'num_attention_heads': 32,
    'num_hidden_layers':32,
    'dropout': 0.1,
    'max_length': 1024,
    'position_embedding': 'learned'
}

class BERT(nn.Module):
    """
    BERT model.

    Args:
        config: Dictionary containing the configuration of the model.

    Attributes:
        encoder: BERTEncoder object
        mlm: MaskedLM object

    It is used for Masked Language Modeling (MLM).
    Here, we use the pretraining mode of the BERTEncoder.
    
    """
    def __init__(
            self,
            vocab_size: int,
            hidden_size: int = 768,
            num_attention_heads: int = 12,
            num_hidden_layers: int = 6,
            dropout: float = 0.1,
            max_length: int = 512,
            position_embedding: str = 'sinusoidal'
        ) -> None:
        
        super(BERT, self).__init__()
        self.encoder = BERTEncoder(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            dropout=dropout,
            max_length=max_length,
            position_embedding=position_embedding
        )
        self.mlm = MaskedLanguageModel(vocab_size, hidden_size)



    def forward(self, input_ids: torch.Tensor, output_attentions = False) -> typing.Tuple[torch.Tensor, typing.List[torch.Tensor], typing.List[torch.Tensor]]:
        encoder_output = self.encoder(input_ids, need_weights = output_attentions)
        if output_attentions:
            x, attention_weights_list, embeddings_list = encoder_output
            return self.mlm(x), attention_weights_list, embeddings_list
        else:
            x,_,_ = encoder_output
            return self.mlm(x)
    
    def predict(self, input_ids: torch.Tensor) -> torch.Tensor:
        encoder_output = self.encoder(input_ids, need_weights = False)
        x, _, _ = encoder_output
        mlm_Y_hat = self.mlm(x)
        topk_val, topk_idx = torch.topk(torch.softmax(mlm_Y_hat, dim=-1), k=1, dim=-1)
        return topk_idx.squeeze(-1)
    
    @classmethod
    def from_pretrained(cls, model_path: str):
        """
        Load the pretrained model from the model_path

        Args:
            model_path: The path to the pretrained model

        Returns:
            model: The pretrained model
        """

        model = cls(
            **NB_CONFIG
        )
        model.load_state_dict(torch.load(model_path))
        return model

        



class BERTEncoder(nn.Module):
    """
    BERT encoder.

    Args:
        config: Dictionary containing the configuration of the model.

    Attributes:
        vocab_size: Size of the vocabulary
        hidden_size: Size of the hidden layer
        num_attention_heads: Number of attention heads
        num_hidden_layers: Number of hidden layers
        dropout: Dropout probability
        max_length: Maximum length of the sequence
        embedding: Embedding layer
        position_embedding: PositionalEncoding layer
        transformer_blocks: List of TransformerBlock objects
        encoder_forward: Function to be used for forward pass

    This encoder part is used for both downstream tasks and pretraining.

    For downstream tasks, the forward pass returns a list of embeddings and a list of attention weights.
    For pretraining, the forward pass returns the final embedding.
    
    """
    def __init__(
            self,
            vocab_size: int,
            hidden_size: int = 768,
            num_attention_heads: int = 12,
            num_hidden_layers: int = 6,
            dropout: float = 0.1,
            max_length: int = 512,
            position_embedding: str = 'sinusoidal',

        ) -> None:
        super(BERTEncoder, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.max_length = max_length
        self.position_embedding = position_embedding

        self.embedding = BERTEmbedding(
            vocab_size=self.vocab_size,
            embed_size=self.hidden_size,
            dropout=self.dropout,
            max_length=self.max_length,
            position_embedding=self.position_embedding
        )
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=self.hidden_size,
                num_attention_heads=self.num_attention_heads,
                ff_dim=self.hidden_size*4,
                dropout=self.dropout,
            )
            for _ in range(self.num_hidden_layers)
        ])

    def forward(self, input_ids: torch.Tensor, need_weights = None) -> typing.Tuple[torch.Tensor, typing.List[torch.Tensor], typing.List[torch.Tensor]]:
        attention_mask = input_ids.ne(0).unsqueeze(2)
        attention_mask = attention_mask.to(dtype=torch.float32)
        
        x = self.embedding(input_ids)
        encoder_output = self.encoder_forward(x, attention_mask, need_weights = need_weights)
        return encoder_output
    
    def encoder_forward(
            self,
            x: torch.Tensor,
            attention_mask: torch.Tensor,
            need_weights = None
        ) -> typing.Tuple[
            None,
            typing.List[torch.Tensor],
            typing.List[torch.Tensor]
        ]:


        first_layer = x
        embeddings_list = [first_layer.unsqueeze(dim=1)]
        attn_output_weights_list = []
        for i in range(self.num_hidden_layers):
            x, attn_output_weights = self.transformer_blocks[i](x, attention_mask, need_weights = need_weights)
            layer_embed = x
            embeddings_list.append(layer_embed.unsqueeze(dim=1))
            if need_weights:
                attn_output_weights_list.append(attn_output_weights.unsqueeze(dim=1))
        return x, attn_output_weights_list, embeddings_list

class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : An embedding layer for input tokens
        2. PositionalEmbedding : An embedding layer for positional encodings

    """
    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 768,
        dropout: float = 0.1,
        max_length: int = 512,
        position_embedding: str ='sinusoidal'
    ):
        """
        :param vocab_size: vocab_size of total words
        :param embed_size: Embedding size of token embedding
        :param dropout: Dropout rate
        :param max_length: Maximum length of sequence
        """
        super().__init__()
        self.max_length = max_length
        self.token = nn.Embedding(vocab_size, embed_size)
        self.position = PositionalEncoding(embed_size, max_length) if position_embedding == 'sinusoidal' else LearnedPositionalEmbedding(embed_size, max_length)

        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, input_ids):
        x = self.token(input_ids)
        position_embedding = self.position(x)
        x = x + position_embedding
        return self.dropout(x)

class TransformerBlock(nn.Module):
    """
    Transformer block.

    Args:
        hidden_size: Size of the hidden layer
        num_attention_heads: Number of attention heads
        ff_dim: Dimension of the feed-forward layer
        dropout: Dropout probability

    Attributes:
        attention: MultiheadAttention layer
        norm1: LayerNorm layer
        ffn: Feed-forward layer
        norm2: LayerNorm layer
        dropout: Dropout layer
        attention_forward: Function to be used for forward pass

    This transformer block is used for both downstream tasks and pretraining.
    For downstream tasks, the forward pass returns a tuple of the embedding and the attention weights.
    For pretraining, the forward pass returns the final embedding.
    
    """
    def __init__(
            self,
            hidden_size: int = 768,
            num_attention_heads: int = 12,
            ff_dim: int = 3072,
            dropout: float = 0.1,
        ) -> None:
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_attention_heads, batch_first = True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, hidden_size)
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor, need_weights = None) -> typing.Tuple[torch.Tensor, typing.Optional[torch.Tensor]]:

        return self.attention_forward(x, attention_mask, need_weights = need_weights)
        
    
    def attention_forward(self, x: torch.Tensor, attention_mask: torch.Tensor, need_weights = None) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        attn_output, attn_output_weights = self.attention(
                                                    x + attention_mask,
                                                    x + attention_mask,
                                                    x + attention_mask,
                                                    need_weights = need_weights,
                                                    average_attn_weights = False
                                                )
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        feed_forward_output = self.ffn(x)
        x = x + self.dropout(feed_forward_output)
        x = self.norm2(x)

        return x, attn_output_weights

class PositionalEncoding(nn.Module):
    """
    Positional encoding.

    Args:
        d_model: Size of the hidden layer
        dropout: Dropout probability
        max_length: Maximum length of the sequence

    Attributes:
        dropout: Dropout layer
        pe: Positional encoding

    This provides sinusoidal positional encoding.
    
    """

    def __init__(
            self,
            d_model: int,
            max_len: int = 5000
        ):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0), :]
        return x

    
class LearnedPositionalEmbedding(nn.Module):
    def __init__(
            self,
            d_model: int,
            max_length: int
        ) -> None:
        super(LearnedPositionalEmbedding, self).__init__()
        self.pembd = nn.Embedding(max_length, d_model)
        self.max_length = max_length
        self.pe = self.pembd.weight

    def forward(self, x):
        pos = torch.arange(x.size(1)).long()
        pos = pos.unsqueeze(0).expand(x.size(0), -1)  # Repeat for each item in the batch
        pos = pos.to(x).long()
        x = x + self.pembd(pos)
        return x
    


class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(
            self,
            vocab_size: int,
            hidden_size: int
        ) -> None:
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        return self.linear(x)



if __name__ == '__main__':
    model = ContactPredictionReg(64,1,32)
    model1 = ContactPredictionReg2()
    print(model, model1)