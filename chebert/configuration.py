from __future__ import annotations

from typing import Union, Dict
from chebert.constants import LOGGER, CONFIG_NAME

import os
import copy
import json


class AlbertConfig(object):
    """
    Tất cả thông tin cấu hình của Albert.
    Khai báo các tham số cấu hình cho Albert và các phương thức loading/downloading/saving các thông tin cấu hình này.
    """
    def __init__(self,
                 vocab_size: int,
                 embedding_size: int = 128,
                 hidden_size: int = 4096,
                 num_hidden_layer: int = 12,
                 num_hidden_groups: int = 1,
                 num_attention_heads: int = 64,
                 intermediate_size: int = 16384,
                 inner_group_num: int = 1,
                 hidden_act: str = "gelu_new",
                 hidden_dropout_prob: float = 0.0,
                 attention_probs_dropout_prob: float = 0.0,
                 max_position_embeddings: int = 512,
                 type_vocab_size: int = 2,
                 initializer_range: float = 0.02,
                 layer_norm_eps: float = 1e-12):
        """
        Parameters:
            :param vocab_size: (`int`, *required*)
                Kích thước từ điển (Vocabulary) của Albert. Định nghĩa số lượng sub-word khác nhau có thể được biểu diễn
                của chuỗi đầu vào.
            :param embedding_size: (`int`, *optional*, defaults to 128)
                Kích thước của các vector embedding.
            :param hidden_size: (`int`, *optional*, defaults to 4096)
                Kích thước của các vector đặc trưng của encoder layer và pooler layer.
            :param num_hidden_layer:
            :param num_hidden_groups:
            :param num_attention_heads:
            :param intermediate_size:
            :param inner_group_num:
            :param hidden_act:
            :param hidden_dropout_prob: (`float`, *required*, defaults to 0.0)
                Giá trị tỉ lệ dropout cho các fully connected layers trong các tầng embeddings, encoder, pooler.
            :param attention_probs_dropout_prob:
            :param max_position_embeddings: (`int`, *required*,  defaults to 512)
                Độ dài tối đa mà mô hình có thể biểu diễn cho chuỗi đầu vào. Thông thường, giá trị thường được thiệt lập
                là 512 hoặc 1024 hoặc 2048.
            :param type_vocab_size: (`int`, *required*, defaults to 2)
                Kích thước của từ điển dùng cho segment embedding.
            :param initializer_range:
            :param layer_norm_eps: (`float`, *required*, defaults to 1e-12)
                Giá trị epsilon sử dụng cho các tầng LayerNorm.
        """
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_hidden_layer = num_hidden_layer
        self.num_hidden_groups = num_hidden_groups
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.inner_group_num = inner_group_num
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

    @classmethod
    def from_pretrained(cls, pretrained_directory: Union[str, os.PathLike]) -> AlbertConfig:
        """
        Khởi tạo AlbertConfig từ file cấu hình tham số đã lưu trước đó.
        Parameters:
            :param pretrained_directory: (`str` or `os.PathLike`)
                Đường dẫn đến thư mục chưa file cấu hình tham số.
            :return: Một instance của AlbertConfig
        """
        albert_config = AlbertConfig(vocab_size=0)

        if os.path.isfile(pretrained_directory):
            raise AssertionError(f"Provided path `{pretrained_directory}` should be a directory, not a file!")

        pretrained_file_path = os.path.join(pretrained_directory, CONFIG_NAME)

        if not os.path.exists(pretrained_file_path):
            raise AssertionError(f"Config file `{pretrained_file_path}` not exists!")

        config_dict = cls.from_json_file(pretrained_file_path)

        for (key, value) in config_dict.items():
            albert_config.__dict__[key] = value
        return albert_config

    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike]) -> Dict:
        """
        Constructs a `AlbertConfig` from a json file of parameters.
        Parameters:
            :param json_file: (`str` or `os.PathLike`)
                Đường dẫn đến file Json cấu hình tham số.
        """
        with open(json_file, "r") as reader:
            text = reader.read()
        return json.loads(text)

    def save_pretrained(self, save_directory: Union[str, os.PathLike]):
        """
        Lưu các thông số cấu hình vào thư mục `save_directory` với định dạng JSON string. Các thông số này có thể được
        re-loaded thông qua phương thức của class `AlbertConfig.from_pretrained`.
        Parameters:
            :param save_directory: (`str` or `os.PathLike`)
                Đường dẫn đến thư mục lưu các thông số cấu hình của Albert. Thư mục sẽ được tạo tự động nếu nó không tồn
                tại.
        """
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file!")

        os.makedirs(save_directory, exist_ok=True)

        save_file_path = os.path.join(save_directory, CONFIG_NAME)

        self.to_json_file(save_file_path)
        LOGGER.info(f"Albert Configuration saved in {save_file_path}")

    def to_dict(self):
        """
        Serializes instance hiện tại thành một Python dictionary.
        """
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self) -> str:
        """
        Serializes instance hiện tại thành một JSON string.
        """
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Lưu instance hiện tại vào json file.
        parameters:
            :param json_file_path: (`str` or `os.PathLike`)
                Đường dẫn đến file lưu các thông số cấu hình của Albert.
        """
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())


# DEBUG
if __name__ == "__main__":
    config = AlbertConfig(vocab_size=30000)
    config.save_pretrained("tmp")
    config = AlbertConfig.from_pretrained("tmp")
    LOGGER.info(config.to_dict())
