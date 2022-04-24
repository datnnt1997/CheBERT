import torch.nn as nn

import torch


class EmbeddingLayer(nn.Module):
    """
    Input represention layer của Albert. Bao gồm TOKEN , POSITION, SEGMENT embedding.
    """
    def __init__(self,
                 vocab_size: int,
                 embedding_size: int,
                 max_position_embeddings: int,
                 type_vocab_size: int,
                 layer_norm_eps: float,
                 hidden_dropout_prob: float):
        """
        Parameters
            :param vocab_size: (`int`, *required*)
                Kích thước từ điển (Vocabulary) của Albert. Định nghĩa số lượng sub-word khác nhau có thể được biểu diễn
                của chuỗi đầu vào.
            :param embedding_size: (`int`, *required*)
                Kích thước của các vector embedding.
            :param max_position_embeddings: (`int`, *required*)
                 Độ dài tối đa mà mô hình có thể biểu diễn cho chuỗi đầu vào. Thông thường, giá trị thường được thiệt
                 lập là 512 hoặc 1024 hoặc 2048.
            :param type_vocab_size: (`int`, *required*)
                Kích thước của từ điển dùng cho segment embedding.
            :param layer_norm_eps: (`float`, *required*)
                Giá trị epsilon sử dụng cho tầng LayerNorm.
            :param hidden_dropout_prob: (`float`, *required*)
                Giá trị tỉ lệ dropout cho các fully connected layers.
        """
        super(EmbeddingLayer, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_position_embeddings, embedding_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, embedding_size)
        self.layer_norm = nn.LayerNorm(embedding_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None):
        seq_length = input_ids.size(1)

        if position_ids is None:
            position_ids = self.position_ids[:, : seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        tokens_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = tokens_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


# DEBUG
if __name__ == "__main__":
    embeddinglayer = EmbeddingLayer(vocab_size=30000,
                                    embedding_size=128,
                                    max_position_embeddings=512,
                                    type_vocab_size=2,
                                    layer_norm_eps=1e-12,
                                    hidden_dropout_prob=0.0)
    print("Layer Architecture:")
    print(embeddinglayer)
    print("Forward output shape:")
    print(embeddinglayer(input_ids=torch.randint(low=0, high=30000, size=[4, 512])).shape)
