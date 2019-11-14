import torch
import torch.nn as nn

class MaxPool1d(nn.Module):
    def __init__(self, kernel_size, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size

        super().__init__()

    def forward(self, input):
        batch_size = input.size(0)
        input_size = input.size(1)
        length_in = input.size(2)
        length_out = 1 + length_in - self.kernel_size

        h = []
        i = 0
        while i < length_out:
            cut = input[:, :, i:i + self.kernel_size]
            value, _ = torch.max(cut, dim=2)
            assert value.size() == (batch_size, input_size)
            h.append( value )
            i += self.stride

        result = torch.stack(h, dim=2)
        assert result.size() == (batch_size, input_size, length_out)
        return result

batch_size = 3
input_size = 6
length_in = 6
kernel_size = 3
stride = 1

data = torch.randn(batch_size, input_size, length_in)
nn_max_pool_1d = nn.MaxPool1d(kernel_size, stride=stride)
n2_max_pool_1d = MaxPool1d(kernel_size, stride=stride)

h1 = nn_max_pool_1d(data)
h2 = n2_max_pool_1d(data)

print(h1)
print(h2)

assert torch.equal(h1, h2)
