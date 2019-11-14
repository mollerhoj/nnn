import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = torch.Tensor(4 * hidden_size, input_size)
        self.weight_hh = torch.Tensor(4 * hidden_size, hidden_size)

        super().__init__()

    def forward(self, x):
        batch_size = x.size(0)
        assert self.input_size == x.size(1)
        h_prev = torch.zeros(batch_size, hidden_size)
        c_prev = torch.zeros(batch_size, hidden_size)

        w_ii, w_if, w_ig, w_io = torch.split(self.weight_ih, hidden_size, dim=0)
        w_hi, w_hf, w_hg, w_ho = torch.split(self.weight_hh, hidden_size, dim=0)

        assert w_ii.size() == (hidden_size, input_size)
        assert w_hi.size() == (hidden_size, hidden_size)

        input_gate  = torch.sigmoid(x @ torch.t(w_ii) + h_prev @ torch.t(w_hi))
        forget_gate = torch.sigmoid(x @ torch.t(w_if) + h_prev @ torch.t(w_hf))
        cell_gate = torch.tanh(x @ torch.t(w_ig) + h_prev @ torch.t(w_hg))
        output_gate = torch.sigmoid(x @ torch.t(w_io) + h_prev @ torch.t(w_ho))
        cell = forget_gate * c_prev + input_gate * cell_gate
        hidden = output_gate * torch.tanh(cell)

        return hidden, cell

batch_size = 5
input_size = 4
hidden_size = 2
data = torch.randn(batch_size, input_size)

nn_lstm_cell = nn.LSTMCell(input_size, hidden_size, bias=False)
n2_lstm_cell = LSTMCell(input_size, hidden_size)
n2_lstm_cell.weight_ih = nn_lstm_cell.weight_ih.clone()
n2_lstm_cell.weight_hh = nn_lstm_cell.weight_hh.clone()

h1, c1 = nn_lstm_cell(data)
h2, c2 = n2_lstm_cell(data)

assert torch.equal(h1, h2)
assert torch.equal(c1, c2)
