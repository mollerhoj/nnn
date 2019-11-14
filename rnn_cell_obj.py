import torch
import torch.nn as nn

class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        super().__init__()
        self.hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.ih = nn.Linear(input_size, hidden_size, bias=False)


    def forward(self, input):
        batch_size = input.size(0)
        h_prev = torch.zeros(batch_size, hidden_size)
        h = torch.tanh(self.ih(input) + self.hh(h_prev))

        return h

batch_size = 5
input_size = 4
hidden_size = 2
data = torch.randn(batch_size, input_size)
nn_rnn = nn.RNNCell(input_size, hidden_size, bias=False)
n2_rnn = RNNCell(input_size, hidden_size)

n2_rnn.w_ih = nn_rnn.weight_ih
n2_rnn.w_hh = nn_rnn.weight_hh

n2_rnn.hh.weight = torch.nn.Parameter(nn_rnn.weight_hh.clone())
n2_rnn.ih.weight = torch.nn.Parameter(nn_rnn.weight_ih.clone())

h1 = nn_rnn(data)
h2 = n2_rnn(data)

print(h1)
print(h2)

assert torch.equal(h1, h2)
