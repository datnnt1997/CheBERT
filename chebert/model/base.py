from typing import Union
from chebert.configuration import AlbertConfig
from chebert.constants import LOGGER, CHECKPOINT_NAME
from chebert.model.modules import EmbeddingLayer, EncoderStack, PoolerLayer

import os
import torch
import torch.nn as nn


class AlbertBase(nn.Module):
    def __init__(self, config):
        super(AlbertBase, self).__init__()

        if not isinstance(config, AlbertConfig):
            raise ValueError(f"Parameter `config` should be an instance of class `AlbertConfig`!")

        self.config = config
        self.embeddings = EmbeddingLayer(config.vocab_size, config.embedding_size, config.max_position_embeddings,
                                         config.type_vocab_size, config.layer_norm_eps, config.hidden_dropout_prob)
        self.encoders = EncoderStack(config.embedding_size, config.hidden_size, config.num_attention_heads,
                                    config.layer_norm_eps, config.intermediate_size, config.hidden_act,
                                    config.hidden_dropout_prob, config.attention_probs_dropout_prob,
                                    config.num_hidden_groups)
        self.pooler = PoolerLayer(config.hidden_size)

        self.apply(self._init_weights)

    @classmethod
    def from_pretrained(cls, pretrained_directory):
        return NotImplementedError

    def save_pretrained(self, save_directory: Union[str, os.PathLike]):
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory!")

        # Save model's configuration
        self.config.save_pretrained(save_directory)

        # Save model state dict
        save_file_path = os.path.join(save_directory, CHECKPOINT_NAME)
        torch.save(self.state_dict(), save_file_path)
        LOGGER.info(f"Albert checkpoint saved in {save_file_path}")

    def _init_weights(self, module):
        """
        Khởi tạo các giá trị ban đâu cho các weight của mô hình.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # [batch_size x 1 x 1 x Seq_length]
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layer

        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        encoder_outputs = self.encoders(embedding_output,
                                        extended_attention_mask,
                                        head_mask=head_mask)
        sequence_output = encoder_outputs
        pooled_output = self.pooler(encoder_outputs)

        outputs = (sequence_output, pooled_output,)
        return outputs


# DEBUG
if __name__ == "__main__":
    model = AlbertBase(config=AlbertConfig(vocab_size=30000))
    print("Layer Architecture:")
    print(model)
    print("Forward output shape:")
    input_ids = torch.randint(low=0, high=30000, size=[4, 128])
    sequence_output, pooled_output = model(input_ids)
    print(f"Sequence output: {sequence_output.shape}")
    print(f"Pooled output: {pooled_output.shape}")
