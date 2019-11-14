import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.w_ih = torch.Tensor(hidden_size, input_size)
        self.w_hh = torch.Tensor(hidden_size, hidden_size)

        super().__init__()

    def forward(self, input):
        length = input.size(0)
        batch_size = input.size(1)

        h = []
        h_prev = torch.zeros(batch_size, hidden_size)
        for x in input:
            h_prev = torch.tanh((x @ torch.t(self.w_ih)) + (h_prev @ torch.t(self.w_hh)))
            h.append(h_prev)

        return torch.stack(h, dim=0), h[length-1]

length = 3
batch_size = 5
input_size = 4
hidden_size = 2
data = torch.randn(length, batch_size, input_size)
nn_rnn = nn.RNN(input_size, hidden_size, bias=False)
n2_rnn = RNN(input_size, hidden_size)
n2_rnn.w_ih = nn_rnn.weight_ih_l0
n2_rnn.w_hh = nn_rnn.weight_hh_l0
h1, _ = nn_rnn(data)
h2, _ = n2_rnn(data)

assert torch.equal(h1, h2)
