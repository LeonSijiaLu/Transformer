"""
This code is designed to implement the Transformer model as described in the paper
"Attention is All You Need."

Materials that helped me:
    https://www.youtube.com/watch?v=bCz4OMemCcA
    https://github.com/hkproj/pytorch-transformer
    https://www.youtube.com/watch?v=kCc8FmEb1nY
    https://github.com/karpathy/nanoGPT
    https://github.com/lilianweng/transformer-tensorflow

Author: Sijia Lu (leonsijialu1@gmail.com)
        Aug 2024
"""

import torch
import torch.nn as nn
import math

class Embedding(nn.Module):
    def __init__(self, d_embed, vocab_size):
        super().__init__()
        self.d_embed = d_embed
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_embed)
    def forward(self, x):
        # scale the embedding according to the paper
        return self.embedding(x) * math.sqrt(self.d_embed)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_embed, seq_len, drop_rate):
        super().__init__()
        self.d_embed = d_embed
        self.seq_len = seq_len
        self.dropout = nn.Dropout(drop_rate)

        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        pe = torch.zeros(seq_len, d_embed)
        div_term = 10000.0 ** (torch.arange(0, d_embed, 2).float() / d_embed)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, max_length, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # fixed, no need to learn grad
        return self.dropout(x)
class FeedForward(nn.Module):
    def __init__(self, d_embed, d_ff, drop_rate):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_embed, d_ff),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(d_ff, d_embed),
        )

    def forward(self, x):
        return self.net(x)
class ResidualConnection(nn.Module):
    def __init__(self, features: int, drop_rate: float):
        super().__init__()
        self.dropout = nn.Dropout(drop_rate)
    def forward(self, x):
        return x + self.dropout(x)
class LayerNormalization(nn.Module):
    def __init__(self, d_embed, bias):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d_embed))
        self.b = nn.Parameter(torch.zeros(d_embed)) if bias else None
        self.eps = 1e-5

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.w * (x - mean) / (std + self.eps) + self.b
class ScaledDotProductAttention(nn.Module):
    def __init__(self, head_size, drop_rate):
        super().__init__()
        self.head_size = head_size
        self.drop_out = nn.Dropout(drop_rate)

    def forward(self, Q, K, V, mask):
        """
        Args:
            Q (query): of size (B, T, C)
            K (key): of size (B, T, C)
            V (value): of size (B, T, C)

            (B, T, C) = (Batch, Time/Sequence length, Channel/d_embed)
        """
        assert self.head_size == Q.shape[-1] == K.shape[-1] == V.shape[-1]
        assert Q.shape[-2] == K.shape[-2] == V.shape[-2]
        assert Q.shape[-3] == K.shape[-3] == V.shape[-3]

        B, T, C = Q.shape[-3], Q.shape[-2], Q.shape[-1]

        attention = self.Q & self.K.transpose(-2, -1) * self.d_embed ** -0.5 # (B, T, T)
        assert (attention.shape[0] == B and attention.shape[1] == T and attention.shape[2] == T), "invalid data format"

        if mask is not None:
            # only access current and history positions, not to any future positions
            attention = attention.masked_fill(mask[:T, :T] == 0, float('-inf'))

        # get the probability indicating how much attention that token should pay to every other token
        # attention[b, i, j]: attention i pays to j
        attention = nn.functional.softmax(attention, dim=-1)

        # dropout to reduce overfitting
        attention = self.drop_out(attention)

        output = attention @ V # (B, T, C)
        assert (output.shape[0] == B and output.shape[1] == T and output.shape[2] == C), "invalid attention data"
        return output
class MultiHeadAttention(nn.Module):
    def __init__(self, d_embed, head_size, drop_rate):
        super().__init__()
        assert d_embed % head_size == 0, "d_embed is not divisible by head_size"

        self.d_embed = d_embed
        self.head_size = head_size
        self.drop_rate = drop_rate

        self.w_q = nn.Linear(in_features=d_embed, out_features=d_embed)
        self.w_k = nn.Linear(in_features=d_embed, out_features=d_embed)
        self.w_v = nn.Linear(in_features=d_embed, out_features=d_embed)
        self.w_o = nn.Linear(in_features=d_embed, out_features=d_embed)

    def forward(self, query, memory, mask):
        """
        Args:
            query (query): of size (B, T, C)
            memory (key / the result of encoder): of size (B, T, C)

            (B, T, C) = (Batch, Time/Sequence length, Channel/d_embed)
        """
        assert query.size(-1) == self.d_embed

        if memory is None:
            memory = query

        Q = self.w_q(query)
        K = self.w_k(memory)
        V = self.w_v(memory)

        # dimension is (self.n_heads * B, T, self.head_size(C//self.n_heads))
        Q_ = torch.concat(torch.split(Q, self.head_size, dim=-1), dim=0)
        K_ = torch.concat(torch.split(K, self.head_size, dim=-1), dim=0)
        V_ = torch.concat(torch.split(V, self.head_size, dim=-1), dim=0)

        concat_mask = None
        if mask is not None:
            concat_mask = mask.repeat(self.head_size, 1, 1)

        # (self.n_heads * B, T, self.head_size)
        sdpa = ScaledDotProductAttention(head_size=self.head_size, drop_rate=self.drop_rate)
        attentions = sdpa.forward(Q=Q_, K=K_, V=V_, mask=concat_mask)

        # (B, T, C)
        B, T, C = Q.size()
        attentions = attentions.contiguous().view(B, T, C)

        return self.w_o(attentions)

class Encoder(nn.Module):
    def __init__(self,
                 multi_head_attention: MultiHeadAttention,
                 feed_forward: FeedForward,
                 d_embed: int,
                 drop_rate: float):
        super().__init__()
        self.multi_head_layer = multi_head_attention
        self.feed_forward = feed_forward
        self.residual = nn.ModuleList([ResidualConnection(features=d_embed, drop_rate=drop_rate) for _ in range(2)])
        self.norm = nn.ModuleList([LayerNormalization(d_embed=d_embed) for _ in range(2)])

    def forward(self, x, source_mask):
        x = self.norm[0](x)
        x = self.multi_head_layer(query=x, memory=x, mask=source_mask)
        x = self.residual[0](x)

        x = self.norm[1](x)
        x = self.feed_forward(x)
        x = self.residual[1](x)
        return x

class Encoders(nn.Module):
    def __init__(self, encoders: nn.ModuleList):
        super().__init__()
        self.encoders = encoders

    def forward(self, x, mask):
        for e in self.encoders:
            x = e(x, mask)
        return x
class Decoder(nn.Module):
    def __init__(self,
                 multi_head_attention: MultiHeadAttention,
                 cross_attention: MultiHeadAttention,
                 feed_forward: FeedForward,
                 d_embed: int,
                 drop_rate: float):
        super().__init__()
        self.d_embed = d_embed

        self.multi_head_attention = multi_head_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward

        self.residual = nn.ModuleList([ResidualConnection(features=d_embed, drop_rate=drop_rate) for _ in range(3)])
        self.norm = nn.ModuleList([LayerNormalization(d_embed=d_embed) for _ in range(3)])

    def forward(self, x, encoder_output, source_mask, target_mask):
        x = self.norm[0](x)
        x = self.multi_head_layer(query=x, memory=x, mask=target_mask)
        x = self.residual[0](x)

        x = self.norm[1](x)
        x = self.cross_attention(query=x, memory=encoder_output, mask=source_mask)
        x = self.residual[1](x)

        x = self.norm[2](x)
        x = self.feed_forward(x)
        x = self.residual[2](x)

        return x

class Decoders(nn.Module):
    def __init__(self, decoders: nn.ModuleList):
        super().__init__()
        self.decoders = decoders

    def forward(self, x, encoder_output, source_mask, target_mask):
        for d in self.decoders:
            x = d(x, encoder_output, source_mask, target_mask)
        return x
class Projection(nn.Module):
    def __init__(self, d_embed, vocab_size):
        super().__init__()
        self.linear = nn.Linear(d_embed, vocab_size)

    def forward(self, x):
        return self.linear(x)
class Transformer(nn.Module):
    def __init__(self,
                 encoders: Encoders,
                 decoders: Decoders,
                 input_embedding: Embedding,
                 input_pos_embedding: PositionalEmbedding,
                 output_embedding: Embedding,
                 output_pos_embedding: PositionalEmbedding,
                 projection: Projection,
                 ):
        super().__init__()
        self.encoders = encoders
        self.decoders = decoders
        self.input_embedding = input_embedding
        self.input_pos_embedding = input_pos_embedding
        self.output_embedding = output_embedding
        self.output_pos_embedding = output_pos_embedding
        self.projection = projection

    def encode(self, src, src_mask):
        src = self.input_embedding(src)
        src = src + self.input_pos_embedding(src)
        return self.encoders(src, src_mask)

    def decode(self, encoder_output, source_mask, target, target_mask):
        target = self.output_embedding(target)
        target = target + self.output_pos_embedding(target)
        return self.decoders(target, encoder_output, source_mask, target_mask)

    def project(self, x):
        return self.projection(x)

def build_transformer(
        src_vocab_size: int,
        src_seq_len: int,
        target_vocab_size: int,
        target_seq_len: int,
        d_embed: int=512,
        d_ff: int=2048,
        drop_rate: float=0.1,
        n_heads: int=8,
        n_copy: int=6,
):
    source_embedding = Embedding(d_embed=d_embed, vocab_size=src_vocab_size)
    source_pos_embedding = PositionalEmbedding(d_embed=d_embed, seq_len=src_seq_len, drop_rate=drop_rate)

    target_embedding = Embedding(d_embed=d_embed, vocab_size=target_vocab_size)
    target_pos_embedding = PositionalEmbedding(d_embed=d_embed, seq_len=target_seq_len, drop_rate=drop_rate)

    encoders = nn.ModuleList()
    for _ in range(n_copy):
        multi_head_attention = MultiHeadAttention(d_embed=d_embed, head_size=d_embed//n_heads, drop_rate=drop_rate)
        feed_forward = FeedForward(d_embed=d_embed, d_ff=d_ff, drop_rate=drop_rate)
        encoder = Encoder(multi_head_attention=multi_head_attention,
                          feed_forward=feed_forward,
                          d_embed=d_embed,
                          drop_rate=drop_rate)
        encoders.append(encoder)

    decoders = nn.ModuleList()
    for _ in range(n_copy):
        multi_head_attention = MultiHeadAttention(d_embed=d_embed, head_size=d_embed // n_heads, drop_rate=drop_rate)
        cross_attetion = MultiHeadAttention(d_embed=d_embed, head_size=d_embed // n_heads, drop_rate=drop_rate)
        feed_forward = FeedForward(d_embed=d_embed, d_ff=d_ff, drop_rate=drop_rate)
        decoder = Decoder(multi_head_attention=multi_head_attention,
                          cross_attention=cross_attetion,
                          feed_forward=feed_forward,
                          d_embed=d_embed,
                          drop_rate=drop_rate)
        decoders.append(decoder)

    projection = Projection(d_embed=d_embed, vocab_size=target_vocab_size)

    transformer = Transformer(encoders=encoders,
                              decoders=decoders,
                              input_embedding=source_embedding,
                              input_pos_embedding=source_pos_embedding,
                              output_embedding=target_embedding,
                              output_pos_embedding=target_pos_embedding,
                              projection=projection)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p) # init weights, so we start with weights that are not extreme values.

    return transformer