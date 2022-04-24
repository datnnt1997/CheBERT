from chebert.model.modules.transformer import TransformerEncoder

import torch.nn as nn


class EncoderStack(nn.Module):
    def __init__(self,
                 embedding_size: int,
                 hidden_size: int,
                 num_attention_heads: int,
                 layer_norm_eps: float,
                 intermediate_size: int,
                 hidden_act: str,
                 hidden_dropout_prob: float,
                 attention_probs_dropout_prob: float,
                 num_hidden_groups: int):
        super(EncoderStack, self).__init__()
        self.input_decomposition = nn.Linear(embedding_size, hidden_size)
        self.encoders = nn.ModuleList([
            TransformerEncoder(num_attention_heads,
                               hidden_size,
                               layer_norm_eps,
                               intermediate_size,
                               hidden_act,
                               hidden_dropout_prob,
                               attention_probs_dropout_prob)
            for _ in range(num_hidden_groups)])

    def forward(self, hidden_states, attention_mask, head_mask):
        hidden_states = self.input_decomposition(hidden_states)
        for layer_idx in range(len(self.encoders)):
            hidden_states = self.encoders[layer_idx](hidden_states, attention_mask, head_mask[layer_idx])
        return hidden_states


# DEBUG
if __name__ == "__main__":
    import torch
    encoder = EncoderStack(embedding_size=128, hidden_size=768, num_attention_heads=12, layer_norm_eps=1e-12,
                                 intermediate_size=3072, hidden_act="gelu_new", hidden_dropout_prob=0.1,
                                 attention_probs_dropout_prob=0.1, num_hidden_groups=1)
    print("Layer Architecture:")
    print(encoder)
    print("Forward output shape:")
    hidden_states = torch.rand([4, 128, 128])
    attention_mask = torch.ones([4, 1, 1, 128])
    attention_mask = (1.0 - attention_mask) * -10000.0
    head_mask = [None] * 12
    print(encoder(hidden_states, attention_mask, head_mask).shape)

