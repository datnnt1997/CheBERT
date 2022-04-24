
import torch.nn as nn


class PoolerLayer(nn.Module):
    def __init__(self,
                 hidden_size: int):
        """
        Parameters:
            :param hidden_size: (`int`, *optional*)
               Kích thước của các vector đặc trưng của pooler layer.
        """
        super(PoolerLayer, self).__init__()

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# DEBUG
if __name__ == "__main__":
    import torch
    embeddinglayer = PoolerLayer(hidden_size=4096)
    print("Layer Architecture:")
    print(embeddinglayer)
    print("Forward output shape:")
    print(embeddinglayer(hidden_states=torch.rand(size=[4, 512, 4096])).shape)
