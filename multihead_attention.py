import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Calculating Attention:
# 1)
#   Encoder: Create K, V, Q as embedding of the source sentence (sentence_length x head_word_embed)
#   Decoder: Create K, V as embedding of the encoder output, Q as the previous decoder layer
# 2)
#   For each head, calculate attention_score for each word in QUERY to each word in KEY ( source_sentence_length, target_sentence_length)
# 3)
#   Multiply attention_score with the VALUE to get the weighted values
# 4)
#   Stack the results of the heads together to reduce dimensions (head_word_embed * num_heads = embed_dim)
# 5)
#   Also return attn_output_weights as a sum of the heads (rather than a stack) because it's easier to visualize

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.out_proj_weight = nn.Parameter(torch.Tensor(self.embed_dim, self.embed_dim))
        self.out_proj_bias = nn.Parameter(torch.Tensor(batch_size, self.embed_dim)) #TODO: batch size used

        # FYI: Does not currently support bias for Q, K and V.
        self.q_linear = nn.Linear(self.embed_dim, self.embed_dim, False)
        self.k_linear = nn.Linear(self.embed_dim, self.embed_dim, False)
        self.v_linear = nn.Linear(self.embed_dim, self.embed_dim, False)
        self.out_linear = nn.Linear(self.embed_dim, self.embed_dim, True)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)


    def forward(self, query, key, value):
        # Assign variables, assert
        assert key.size() == value.size()
        target_len = query.size(0)
        batch_size = query.size(1)
        assert self.head_dim * self.num_heads == self.embed_dim

        # Linear projections of Q, K and V
        q = self.q_linear(query)
        assert q.size() == (target_len, batch_size, self.embed_dim)
        k = self.k_linear(key)
        v = self.v_linear(value)

        ## Scale q
        scaling = math.sqrt(1/float(self.head_dim))
        q = q * scaling

        # Split into heads
        q = q.contiguous().view(target_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        source_len = k.size(1)
        v = v.contiguous().view(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)

        # calculate (q @ k) attention weights (attn_w)
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert attn_output_weights.size() == (batch_size * self.num_heads, target_len, source_len)

        attn_output_weights = self.softmax(attn_output_weights)
        attn_output_weights = self.dropout(attn_output_weights)

        # Use attention weights on values (attn_w @ v)
        attn_output = torch.bmm(attn_output_weights, v)
        assert attn_output.size() == (batch_size * self.num_heads, target_len, self.head_dim)

        attn_output = attn_output.transpose(0, 1).contiguous().view(target_len, batch_size, self.embed_dim)

        # Linear out-projection
        attn_output = self.out_linear(attn_output)

        # Return value
        attn_output_weights = attn_output_weights.view(batch_size, self.num_heads, target_len, source_len)
        return attn_output, attn_output_weights.sum(dim=1) / self.num_heads

training = True

embed_size = 4
batch_size = 3
target_len = 6
source_len = 5
num_heads = 2

query = torch.randn(target_len, batch_size, embed_size)
key = torch.randn(source_len, batch_size, embed_size)
value = torch.randn(source_len, batch_size, embed_size)

q_proj_weight = torch.randn(embed_size, embed_size)
k_proj_weight = torch.randn(embed_size, embed_size)
v_proj_weight = torch.randn(embed_size, embed_size)

out_proj_weight = torch.randn(embed_size, embed_size)
out_proj_bias = torch.zeros(batch_size, embed_size)

multi_head_attention = MultiheadAttention(query.size(2), num_heads, dropout=0.0)
multi_head_attention.training = training
multi_head_attention.q_linear.weight = nn.Parameter(q_proj_weight.clone())
multi_head_attention.k_linear.weight = nn.Parameter(k_proj_weight.clone())
multi_head_attention.v_linear.weight = nn.Parameter(v_proj_weight.clone())

multi_head_attention.out_linear.weight = nn.Parameter(out_proj_weight.clone())
multi_head_attention.out_linear.bias = nn.Parameter(out_proj_bias.clone())

# for p, n in multi_head_attention.named_parameters():
#     print(p, n)

attn_output1, attn_output_weights1 = multi_head_attention.forward(query, key, value)

attn_output2, attn_output_weights2 = F.multi_head_attention_forward(query, key, value, query.size(2), num_heads, None, None, None, None, False, 0.0, out_proj_weight, out_proj_bias, training, None, True, None, True, q_proj_weight, k_proj_weight, v_proj_weight)

assert torch.equal(attn_output1, attn_output2)
assert torch.equal(attn_output_weights1, attn_output_weights2)

