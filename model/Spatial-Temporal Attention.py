import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Spatial_Attention(nn.Module):
    def __init__(self, d_k):
        super(Spatial_Attention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)
        return scores

class Temporal_Attention(nn.Module):
    def __init__(self, d_k, num_of_d):
        super(Temporal_Attention, self).__init__()
        self.d_k = d_k
        self.num_of_d =num_of_d

    def forward(self, Q, K, V, attn_mask, res_att):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) + res_att
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)
        attn = F.softmax(scores, dim=3)
        context = torch.matmul(attn, V)
        return context, scores

class SMultiHeadAttention(nn.Module):
    def __init__(self, DEVICE, d_model, d_k ,d_v, n_heads):
        super(SMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.DEVICE = DEVICE
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)

    def forward(self, input_Q, input_K, attn_mask):
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        attn = Spatial_Attention(self.d_k)(Q, K, attn_mask)
        return attn

class MultiHeadAttention(nn.Module):
    def __init__(self, DEVICE, d_model, d_k ,d_v, n_heads, num_of_d):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.num_of_d = num_of_d
        self.DEVICE = DEVICE
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask, res_att):
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, self.num_of_d, -1, self.n_heads, self.d_k).transpose(2, 3)
        K = self.W_K(input_K).view(batch_size, self.num_of_d, -1, self.n_heads, self.d_k).transpose(2, 3)
        V = self.W_V(input_V).view(batch_size, self.num_of_d, -1, self.n_heads, self.d_v).transpose(2, 3)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        context, res_attn = Spatial_Attention(self.d_k, self.num_of_d)(Q, K, V, attn_mask, res_att)
        context = context.transpose(2, 3).reshape(batch_size, self.num_of_d, -1, self.n_heads * self.d_v)
        output = self.fc(context)

        return nn.LayerNorm(self.d_model).to(self.DEVICE)(output + residual), res_attn

