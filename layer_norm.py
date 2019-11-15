import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, shape, eps=1e-05):
        super().__init__()
        self.shape = shape
        self.eps = eps

    def forward(self, input):
        batch_size = input.size(0)
        embed_size = input.size(1)

        mean = torch.unsqueeze(input.mean(dim=1), dim=-1)
        assert mean.size() == (batch_size, 1)

        var = torch.unsqueeze(input.var(dim=1, unbiased=False), dim=-1)
        assert var.size() == (batch_size, 1)

        result = ((input - mean) / torch.sqrt(var + self.eps) )

        return result

batch_size = 19
embed_size = 5

data = torch.randn(batch_size, embed_size) * 7 + 7 # (mean 7, std 7)

nn_layer_norm = nn.LayerNorm(embed_size)
nn_layer_norm.training = True
n2_layer_norm = LayerNorm(embed_size)

h1 = nn_layer_norm(data)
h2 = n2_layer_norm(data)

def almost_equal(a, b):
    return torch.all(torch.lt(torch.abs(torch.add(a, -b)), 1e-5))

assert almost_equal(h1, h2)

# Mean becomes 0
assert almost_equal(torch.zeros(batch_size), h1.mean(dim=1))

# Var becomes 1 + 1/(embed_size-1) (because we use unbiased=False, (no bessels correction))
assert almost_equal(torch.ones(batch_size) + 1/(embed_size-1), h1.var(dim=1))

