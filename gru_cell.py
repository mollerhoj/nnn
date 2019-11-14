import torch
import torch.nn as nn

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = torch.Tensor(3 * hidden_size, input_size)
        self.weight_hh = torch.Tensor(3 * hidden_size, hidden_size)

        super().__init__()

    def forward(self, x):
        batch_size = x.size(0)
        assert self.input_size == x.size(1)
        h_prev = torch.zeros(batch_size, hidden_size)
        c_prev = torch.zeros(batch_size, hidden_size)

        w_ir, w_iz, w_in = torch.split(self.weight_ih, hidden_size, dim=0)
        w_hr, w_hz, w_hn = torch.split(self.weight_hh, hidden_size, dim=0)

        assert w_ir.size() == (hidden_size, input_size)
        assert w_hr.size() == (hidden_size, hidden_size)

        # reset, update, new
        r_gate = torch.sigmoid(x @ torch.t(w_ir) + h_prev @ torch.t(w_hr))
        z_gate = torch.sigmoid(x @ torch.t(w_iz) + h_prev @ torch.t(w_hz))
        n_gate = torch.tanh(x @ torch.t(w_in) + r_gate * (h_prev @ torch.t(w_hn)))
        hidden = (1 - z_gate) * n_gate + z_gate * h_prev

        return hidden

batch_size = 5
input_size = 4
hidden_size = 2
data = torch.randn(batch_size, input_size)

nn_gru_cell = nn.GRUCell(input_size, hidden_size, bias=False)
n2_gru_cell = GRUCell(input_size, hidden_size)
n2_gru_cell.weight_ih = nn_gru_cell.weight_ih.clone()
n2_gru_cell.weight_hh = nn_gru_cell.weight_hh.clone()

h1 = nn_gru_cell(data)
h2 = n2_gru_cell(data)

def almost_equal(a, b):
    return torch.all(torch.lt(torch.abs(torch.add(a, -b)), 1e-5))

assert almost_equal(h1, h2)

