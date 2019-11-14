import torch
import torch.nn as nn

class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        super().__init__()
        self.weight = torch.Tensor(out_channels, in_channels, kernel_size)

    def forward(self, input):
        batch_size = input.size(0)
        assert self.in_channels == input.size(1)
        length_in = input.size(2)
        length_out = 1 + length_in - self.kernel_size

        h = []
        i = 0
        while i < length_out:
            cut = input[:, :, i:i + self.kernel_size]
            assert cut.size() == (batch_size, self.in_channels, kernel_size)

            cut = cut.permute(2, 0, 1)
            assert cut.size() == (kernel_size, batch_size, self.in_channels)

            w = self.weight.permute(2, 1, 0)
            assert w.size() == (kernel_size, self.in_channels, self.out_channels)

            values = torch.bmm(cut, w)
            assert values.size() == (kernel_size, batch_size, self.out_channels)

            value = torch.sum(values, dim=0)
            assert value.size() == (batch_size, self.out_channels)

            h.append( value )
            i += self.stride

        result = torch.stack(h, dim=2)

        assert result.size() == (batch_size, self.out_channels, length_out)

        return result

batch_size = 3
in_channels = 2
out_channels = 2
length_in = 6
kernel_size = 3
stride = 1

data = torch.randn(batch_size, in_channels, length_in)
nn_conv_1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, bias=False)
n2_conv_1d = Conv1d(in_channels, out_channels, kernel_size, stride=stride)
n2_conv_1d.weight = nn_conv_1d.weight.clone()

h1 = nn_conv_1d(data)
h2 = n2_conv_1d(data)

def almost_equal(a, b):
    return torch.all(torch.lt(torch.abs(torch.add(a, -b)), 1e-5))

assert almost_equal(h1, h2)
