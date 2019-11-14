import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = torch.randn(out_features, in_features)
        self.bias = torch.randn(1, out_features)

    def forward(self, input):
        return input @ torch.t(self.weight) + self.bias

features = 10
batch_size = 5
data = torch.randn(batch_size, features)
nn_linear = nn.Linear(10,2)
n2_linear = Linear(10,2)
n2_linear.weight = nn_linear.weight
n2_linear.bias = nn_linear.bias

assert torch.equal(n2_linear(data), nn_linear(data))
