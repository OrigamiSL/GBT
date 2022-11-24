import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft

from einops import rearrange, reduce, repeat
import math, random

from .modules import Feedforward
from .exponential_smoothing import ExponentialSmoothing


class GrowthLayer(nn.Module):

    def __init__(self, d_model, nhead, d_head=None, dropout=0.1):
        super().__init__()
        self.d_head = d_head or (d_model // nhead)
        self.d_model = d_model
        self.nhead = nhead

        self.z0 = nn.Parameter(torch.randn(self.nhead, self.d_head))
        self.in_proj = nn.Linear(self.d_model, self.d_head * self.nhead)
        self.es = ExponentialSmoothing(self.d_head, self.nhead, dropout=dropout)
        self.out_proj = nn.Linear(self.d_head * self.nhead, self.d_model)

        assert self.d_head * self.nhead == self.d_model, "d_model must be divisible by nhead"

    def forward(self, inputs):
        """
        :param inputs: shape: (batch, seq_len, dim)
        :return: shape: (batch, seq_len, dim)
        """
        B, T, D = inputs.shape
        values = self.in_proj(inputs).view(B, T, self.nhead, -1)
        values = torch.cat([repeat(self.z0, 'H D -> B () H D', B=B), values], dim=1)
        values = values[:, 1:] - values[:, :-1]
        out = self.es(values)
        out = torch.cat([repeat(self.es.v0, 'H D -> B () H D', B=B), out], dim=1)
        out = rearrange(out, 'B T H D -> B T (H D)')
        return self.out_proj(out)


class FourierLayer(nn.Module):

    def __init__(self, d_model, pred_len, k=None, low_freq=1):
        super().__init__()
        self.d_model = d_model
        self.pred_len = pred_len
        self.k = k
        self.low_freq = low_freq

    def forward(self, x):
        """x: (B T D)"""
        B, T, D = x.shape
        x_freq = fft.rfft(x, dim=1)

        if T % 2 == 0:
            x_freq = x_freq[:, self.low_freq:-1]
            f = fft.rfftfreq(T)[self.low_freq:-1]
        else:
            x_freq = x_freq[:, self.low_freq:]
            f = fft.rfftfreq(T)[self.low_freq:]
        x_freq, index_tuple = self.topk_freq(x_freq)
        f = repeat(f, 'F -> B F D', B=x_freq.size(0), D=x_freq.size(2))
        f = rearrange(f[index_tuple], 'B F D -> B F () D').to(x_freq.device)

        return self.extrapolate(x_freq, f, T)

    def extrapolate(self, x_freq, f, T):
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)
        t = rearrange(torch.arange(T + self.pred_len, dtype=torch.float),
                      'T -> () () T ()').to(x_freq.device)

        amp = rearrange(x_freq.abs() / T, 'B F D -> B F () D')
        phase = rearrange(x_freq.angle(), 'B F D -> B F () D')

        x_time = amp * torch.cos(2 * math.pi * f * t + phase)

        return reduce(x_time, 'B F T D -> B T D', 'sum')

    def topk_freq(self, x_freq):
        values, indices = torch.topk(x_freq.abs(), self.k, dim=1, largest=True, sorted=True)
        mesh_a, mesh_b = torch.meshgrid(torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)))
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        x_freq = x_freq[index_tuple]

        return x_freq, index_tuple


class LevelLayer(nn.Module):

    def __init__(self, d_model, c_out, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.c_out = c_out

        self.es = ExponentialSmoothing(1, self.c_out, dropout=dropout, aux=True)
        self.growth_pred = nn.Linear(self.d_model, self.c_out)
        self.season_pred = nn.Linear(self.d_model, self.c_out)

    def forward(self, level, growth, season):
        B, T, _ = level.shape
        growth = self.growth_pred(growth).view(B, T, self.c_out, 1)
        season = self.season_pred(season).view(B, T, self.c_out, 1)
        growth = growth.view(B, T, self.c_out, 1)
        season = season.view(B, T, self.c_out, 1)
        level = level.view(B, T, self.c_out, 1)
        out = self.es(level - season, aux_values=growth)
        out = rearrange(out, 'B T H D -> B T (H D)')
        return out

class EncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, c_out, pred_len, k, dropout=0.1,
                 activation='sigmoid', layer_norm_eps=1e-5):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.c_out = c_out
        self.pred_len = pred_len
        d_ff = 4 * d_model
        self.d_ff = d_ff

        self.growth_layer = GrowthLayer(d_model, nhead, dropout=dropout)
        self.seasonal_layer = FourierLayer(d_model, pred_len, k=k)
        self.level_layer = LevelLayer(d_model, c_out, dropout=dropout)

        # Implementation of Feedforward model
        self.ff = Feedforward(d_model, d_ff, dropout=dropout, activation=activation)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, res, level, attn_mask=None):
        season = self._season_block(res)
        res = res - season[:, :-self.pred_len]
        growth = self._growth_block(res)
        res = self.norm1(res - growth[:, 1:])
        res = self.norm2(res + self.ff(res))

        level = self.level_layer(level, growth[:, :-1], season[:, :-self.pred_len])

        return res, level, growth, season

    def _growth_block(self, x):
        x = self.growth_layer(x)
        return self.dropout1(x)

    def _season_block(self, x):
        x = self.seasonal_layer(x)
        return self.dropout2(x)


class Encoder(nn.Module):

    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, res, level, attn_mask=None):
        growths = []
        seasons = []
        for layer in self.layers:
            res, level, growth, season = layer(res, level, attn_mask=None)
            growths.append(growth)
            seasons.append(season)

        return level, growths, seasons
