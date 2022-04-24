from chebert.model.modules.utils import gelu, swish, gelu_new
from torch.nn.functional import relu

import math
import torch
import torch.nn as nn


ACT2FN = {"gelu": gelu, "relu": relu, "swish": swish, "gelu_new": gelu_new}


class FeedForwardNetwork(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 hidden_act: str,
                 hidden_dropout_prob: float):
        super(FeedForwardNetwork, self).__init__()
        self.input_dense = nn.Linear(hidden_size, intermediate_size)
        self.activation = ACT2FN[hidden_act]
        self.output_dense = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, attention_output):
        intermediate_output = self.input_dense(attention_output)
        intermediate_output = self.activation(intermediate_output)
        ffn_output = self.output_dense(intermediate_output)
        ffn_output = self.dropout(ffn_output)
        return ffn_output


class SelfAttention(nn.Module):
    def __init__(self,
                 num_attention_heads: int,
                 hidden_size: int,
                 hidden_dropout_prob: float,
                 attention_probs_dropout_prob: float):
        super(SelfAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.query = nn.Linear(hidden_size, self.all_head_size)

        self.output_dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.attention_dropout = nn.Dropout(attention_probs_dropout_prob)

    def __transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        # [Batch_size x Seq_len x Hidden_size]
        mixed_query_layer = self.query(hidden_states)
        # [Batch_size x Seq_len x Hidden_size]
        mixed_key_layer = self.key(hidden_states)
        # [Batch_size x Seq_len x Hidden_size]
        mixed_value_layer = self.value(hidden_states)

        # [Batch_size x Num_of_heads x Seq_len x Head_size]
        query_layer = self.__transpose_for_scores(mixed_query_layer)
        # [Batch_size x Num_of_heads x Seq_len x Head_size]
        key_layer = self.__transpose_for_scores(mixed_key_layer)
        # [Batch_size x Num_of_heads x Seq_len x Head_size]
        value_layer = self.__transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class TransformerEncoder(nn.Module):
    def __init__(self,
                 num_attention_heads: int,
                 hidden_size: int,
                 layer_norm_eps: float,
                 intermediate_size: int,
                 hidden_act: str,
                 hidden_dropout_prob: float,
                 attention_probs_dropout_prob: float):
        super(TransformerEncoder, self).__init__()
        self.self_attention = SelfAttention(num_attention_heads, hidden_size, hidden_dropout_prob,
                                            attention_probs_dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.feed_forward_net = FeedForwardNetwork(hidden_size, intermediate_size, hidden_act, hidden_dropout_prob)
        self.output_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        # Self-Attention
        attention_output = self.self_attention(hidden_states, attention_mask, head_mask)
        # Add & Normalize
        attention_output = self.layer_norm(attention_output + hidden_states)
        # Feed Forward
        ffn_output = self.feed_forward_net(attention_output)
        # Add & Normalize
        output = self.layer_norm(ffn_output)

        return output


#DEBUG
if __name__ == "__main__":
    encoder = TransformerEncoder(num_attention_heads=12, hidden_size=768, layer_norm_eps=1e-12,
                                 intermediate_size=3072, hidden_act="gelu_new", hidden_dropout_prob=0.1,
                                 attention_probs_dropout_prob=0.1)
    print("Layer Architecture:")
    print(encoder)
    print("Forward output shape:")
    hidden_states = torch.rand([4, 128, 128])
    attention_mask = torch.ones([4, 1, 1, 128])
    attention_mask = (1.0 - attention_mask) * -10000.0
    print(encoder(hidden_states, attention_mask, head_mask=None).shape)
