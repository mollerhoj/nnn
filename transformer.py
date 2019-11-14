import torch
import copy
import torch.nn as nn
import torch.nn.functional as F

from multihead_attention import MultiheadAttention

class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, d_hidden=2048, dropout=0.1):
        super(Transformer, self).__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, d_hidden, dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, d_hidden, dropout)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.d_model = d_model
        self.nhead = nhead

    def forward(self, source, target):
        assert target.size(1) == source.size(1) # batch_size must be equal
        assert target.size(2) == source.size(2) # embed_size must be equal

        memory = self.encoder(source)
        result = self.decoder(target, memory)
        return result

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, source):
        for i in range(self.num_layers):
            source = self.layers[i](source)
        return self.norm(source)

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, target, memory):
        for i in range(self.num_layers):
            target = self.layers[i](target, memory)
        return self.norm(target)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_hidden=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, d_hidden)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_hidden, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, source):
        source_tmp = self.self_attn(source, source, source)[0]
        source = source + self.dropout1(source_tmp)
        source = self.norm1(source)

        source_tmp = self.linear2(self.dropout(F.relu(self.linear1(source))))
        source = source + self.dropout2(source_tmp)
        source = self.norm2(source)
        return source

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_hidden=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, d_hidden)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_hidden, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, target, memory):
        target_tmp = self.self_attn(target, target, target)[0]
        target = target + self.dropout1(target_tmp)
        target = self.norm1(target)

        target_tmp = self.multihead_attn(target, memory, memory)[0]
        target = target + self.dropout2(target_tmp)
        target = self.norm2(target)

        target_tmp = self.linear2(self.dropout(F.relu(self.linear1(target))))
        target = target + self.dropout3(target_tmp)
        target = self.norm3(target)
        return target

embed_size = 4
source_len = 10
target_len = 20
batch_size = 3

source = torch.rand((source_len, batch_size, embed_size))
target = torch.rand((target_len, batch_size, embed_size))

nn_transformer = nn.Transformer(d_model=embed_size, nhead=2, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=3, dropout=0.0)
n2_transformer = Transformer(d_model=embed_size, nhead=2, num_encoder_layers=2, num_decoder_layers=2, d_hidden=3, dropout=0.0)

# Initialize our home made transformer with the same weights, so we can compare results:
n2_transformer.load_state_dict(nn_transformer.state_dict())

nn_out = nn_transformer(source, target)
n2_out = n2_transformer(source, target)

assert torch.equal(nn_out, n2_out)
