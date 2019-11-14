import torch
import torch.nn as nn

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048):
        self.d_model = d_model # input features
        self.nhead = nhead     # attention heads
        self.dim_feedforward = dim_feedforward

        super().__init__()

    def forward(self, src):

        assert src.size() == (length_s, batch_size, e)
        # src: (S, N, E)

        return output

batch_size = 5
d_model = 4
nhead = 2
dim_feedforward = 3

data = torch.randn(batch_size, d_model)

nn_transformer_encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=dim_feedforward, dropout=0)
n2_transformer_encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward=dim_feedforward, dropout=0)
h1 = nn_transformer_encoder_layer(data)
h2 = n2_transformer_encoder_layer(data)

def almost_equal(a, b):
    return torch.all(torch.lt(torch.abs(torch.add(a, -b)), 1e-5))

assert almost_equal(h1, h2)

