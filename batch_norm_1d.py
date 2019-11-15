import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchNorm1d(nn.Module):
    def __init__(self, embed_size, eps=1e-05):
        super().__init__()
        self.eps = eps
        self.weight = torch.ones(embed_size)
        self.bias = torch.zeros(embed_size)


    def forward(self, input):
        batch_size = input.size(0)
        embed_size = input.size(1)

        mean = input.mean(dim=0)
        assert mean.size() == (embed_size,)

        var = input.var(dim=0, unbiased=False)
        assert var.size() == (embed_size,)

        result = ((input - mean) / torch.sqrt(var + self.eps) )

        return result

batch_size = 59
embed_size = 40

data = torch.randn(batch_size, embed_size) * 7 + 7 # (mean 7, std 7)

nn_batch_norm_1d = nn.BatchNorm1d(embed_size, momentum=0.0, affine=False)
nn_batch_norm_1d.training = True
n2_batch_norm_1d = BatchNorm1d(embed_size)

h1 = nn_batch_norm_1d(data)
h2 = n2_batch_norm_1d(data)

# TODO: Support momentum
def almost_equal(a, b):
    return torch.all(torch.lt(torch.abs(torch.add(a, -b)), 1e-4))

assert almost_equal(h1, h2)

# Mean becomes 0
assert almost_equal(torch.zeros(embed_size), h1.mean(dim=0))

# Var becomes 1 + 1/(batch_size-1) (because we use unbiased=False, (no bessels correction))
assert almost_equal(torch.ones(embed_size) + 1/(batch_size-1), h1.var(dim=0))

