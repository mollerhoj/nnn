import torch
import torch.nn as nn

class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.w_ih = torch.Tensor(hidden_size, input_size)
        self.w_hh = torch.Tensor(hidden_size, hidden_size)

        super().__init__()

    def forward(self, input):
        batch_size = input.size(0)
        h_prev = torch.zeros(batch_size, hidden_size)
        h = torch.tanh((input @ torch.t(self.w_ih)) + (h_prev @ torch.t(self.w_hh)))

        return h

batch_size = 5
input_size = 4
hidden_size = 2
data = torch.randn(batch_size, input_size)
nn_rnn = nn.RNNCell(input_size, hidden_size, bias=False)
n2_rnn = RNNCell(input_size, hidden_size)

n2_rnn.w_ih = nn_rnn.weight_ih
n2_rnn.w_hh = nn_rnn.weight_hh
h1 = nn_rnn(data)
h2 = n2_rnn(data)

print(h1)
print(h2)

assert torch.equal(h1, h2)
