import numpy as np
import torch
import torch.nn as nn
from Self_Regression.attn import ProbAttention, FullAttention, AttentionLayer
from Self_Regression.decoder import Decoder, DecoderLayer
from Self_Regression.embed import DataEmbedding
import torch.nn.functional as F


class NHITSBlock(nn.Module):
    def __init__(self, seq_len, pred_len,
                 n_layers,
                 n_hidden, theta_size, n_pool_kernel_size,
                 basis_function: nn.Module):
        super().__init__()
        n_theta_hidden = [n_hidden, n_hidden]
        n_time_in_pooled = int(np.ceil(seq_len / n_pool_kernel_size))

        n_theta_hidden = [n_time_in_pooled] + n_theta_hidden
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.pooling_layer = nn.MaxPool1d(kernel_size=n_pool_kernel_size,
                                          stride=n_pool_kernel_size, ceil_mode=True)
        hidden_layers = [nn.Linear(in_features=n_theta_hidden[i], out_features=n_theta_hidden[i + 1])
                                       for i in range(n_layers)]
        self.layers = nn.ModuleList(hidden_layers)

        self.basis_parameters = nn.Linear(in_features=n_theta_hidden[-1], out_features=theta_size)
        self.basis_function = basis_function

    def forward(self, x):
        block_input = x
        block_input = block_input.unsqueeze(1)
        block_input = self.pooling_layer(block_input)
        for layer in self.layers:
            block_input = torch.relu(layer(block_input))
        block_input = block_input.squeeze(1)
        basis_parameters = self.basis_parameters(block_input)
        return self.basis_function(basis_parameters)


class NHiTS(nn.Module):
    def __init__(self, seq_len, pred_len, d_model, dropout, time, attn, factor, mix,
                 activation, d_layers, output_attention, n_heads, stack_num,
                 n_pool_kernel_size, n_layers, n_hidden, n_freq_downsample):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.output_attention = output_attention
        Blocks = []
        for i in range(stack_num):
            n_theta = (seq_len + max(pred_len // n_freq_downsample[i], 1))
            Blocks.append(NHITSBlock(seq_len, pred_len,
                          n_layers[i], n_hidden, n_theta, n_pool_kernel_size[i],
                          basis_function=IdentityBasis(backcast_size=seq_len,
                                                       forecast_size=pred_len)))

        self.blocks = nn.ModuleList(Blocks)
        self.group = 1
        self.n_heads = n_heads

        # Encoding
        print("Start Embedding")
        # decoding
        self.dec_embedding = DataEmbedding(1, d_model, dropout, True, time, group=1)
        print("Embedding finished")
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, self.n_heads, mix=mix, group=self.group),
                    d_model=d_model,
                    dropout=dropout,
                    activation=activation,
                    group=self.group
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=1, kernel_size=1, groups=1,
                                    bias=True)

    def forward(self, batch_x=None, batch_x_mark=None, dec_inp=None, x_mark_dec=None, flag='first stage'):
        x = dec_inp[:, :self.seq_len].squeeze()
        if self.pred_len > self.seq_len:
            x_mask = torch.ones_like(x)
        else:
            x_mask = torch.zeros_like(x)
            x_mask[:, -self.pred_len:] = torch.ones_like(x[:, -self.pred_len:])
        residuals = x.flip(dims=(1,))
        forecast = x[:, -1:]

        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(residuals)
            residuals = (residuals - backcast) * x_mask
            forecast = forecast + block_forecast
        first_stage_out = forecast.unsqueeze(-1)
        if flag == 'first stage':
            return first_stage_out
        elif flag == 'second stage':
            first_stage_out = first_stage_out.clone().detach()
            dec_out = self.dec_embedding(first_stage_out, x_mark_dec[:, self.seq_len:, :])
            dec_out = self.decoder(dec_out)
            dec_out = self.projection(dec_out.permute(0, 2, 1)).transpose(1, 2)
            if self.output_attention:
                return dec_out[:, -self.pred_len:, :] + first_stage_out, _
            else:
                return dec_out[:, -self.pred_len:, :] + first_stage_out  # [B, L, D]



class IdentityBasis(nn.Module):
    """
    Identity basis function.
    """
    def __init__(self, backcast_size, forecast_size):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta):
        backcast = theta[:, :self.backcast_size]
        forecast = theta[:, -self.forecast_size:]
        knots = theta[:, self.backcast_size:]
        knots = knots[:, None, :]
        forecast = F.interpolate(knots, size=self.forecast_size)  # , align_corners=True)
        forecast = forecast[:, 0, :]

        return backcast, forecast
