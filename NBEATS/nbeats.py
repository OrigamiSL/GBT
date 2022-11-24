# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
N-BEATS Model.
"""
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from Self_Regression.embed import DataEmbedding
from FEDformer.Autoformer_EncDec import Auto_Decoder, Auto_DecoderLayer, my_Layernorm, series_decomp, series_decomp_multi
from FEDformer.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer

class NBeatsBlock(nn.Module):
    """
    N-BEATS block which takes a basis function as an argumentorch.
    """
    def __init__(self,
                 input_size,
                 theta_size,
                 basis_function: nn.Module,
                 layers,
                 layer_size):
        """
        N-BEATS block.

        :param input_size: Insample size.
        :param theta_size:  Number of parameters for the basis function.
        :param basis_function: Basis function which takes the parameters and produces backcast and forecastorch.
        :param layers: Number of layers.
        :param layer_size: Layer size.
        """
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(in_features=input_size, out_features=layer_size)] +
                                      [nn.Linear(in_features=layer_size, out_features=layer_size)
                                       for _ in range(layers - 1)])
        self.basis_parameters = nn.Linear(in_features=layer_size, out_features=theta_size)
        self.basis_function = basis_function

    def forward(self, x):
        block_input = x
        for layer in self.layers:
            block_input = torch.relu(layer(block_input))
        basis_parameters = self.basis_parameters(block_input)
        return self.basis_function(basis_parameters)


class NBeats(nn.Module):
    """
    N-Beats Model.
    """
    def __init__(self, seq_len, pred_len, c_out, d_model, dropout, time, attn, factor, mix,
                 activation, d_layers, output_attention, n_heads, moving_avg, trend_blocks,
                 trend_layers, trend_layer_size, degree_of_polynomial, seasonality_blocks,
                 seasonality_layers, seasonality_layer_size, num_of_harmonics, instance):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.output_attention = output_attention
        trend_block = NBeatsBlock(input_size=seq_len,
                                  theta_size=2 * (degree_of_polynomial + 1),
                                  basis_function=TrendBasis(degree_of_polynomial=degree_of_polynomial,
                                                            backcast_size=seq_len,
                                                            forecast_size=pred_len),
                                  layers=trend_layers,
                                  layer_size=trend_layer_size)
        seasonality_block = NBeatsBlock(input_size=seq_len,
                                        theta_size=4 * int(
                                            np.ceil(num_of_harmonics / 2 * pred_len) - (num_of_harmonics - 1)),
                                        basis_function=SeasonalityBasis(harmonics=num_of_harmonics,
                                                                        backcast_size=seq_len,
                                                                        forecast_size=pred_len),
                                        layers=seasonality_layers,
                                        layer_size=seasonality_layer_size)

        Blocks = [trend_block for _ in range(trend_blocks)] + [seasonality_block for _ in range(seasonality_blocks)]
        self.blocks = nn.ModuleList(Blocks)
        self.group = 1
        self.n_heads = n_heads

        if instance:
            enc_in = 1
            dec_in = 1
            c_out = 1

        # Encoding
        print("Start Embedding")
        # decoding
        self.dec_embedding = DataEmbedding(1, d_model, dropout, True, time, group=1)
        print("Embedding finished")
        # Decomp
        kernel_size = moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
            self.decomp2 = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)
            self.decomp2 = series_decomp(kernel_size)
        # Decoder
        self.decoder = Auto_Decoder(
            [
                Auto_DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, factor, attention_dropout=dropout,
                                        output_attention=False),
                        d_model, n_heads),
                    # AutoCorrelationLayer(
                    #     AutoCorrelation(False, factor, attention_dropout=dropout,
                    #                     output_attention=False),
                    #     d_model, n_heads),
                    d_model=d_model,
                    c_out=c_out,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=my_Layernorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True)
        )

    def forward(self, batch_x=None, batch_x_mark=None, dec_inp=None, x_mark_dec=None, flag='first stage'):
        x = dec_inp[:, :self.seq_len, :].squeeze()
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
            seasonal_second, trend_second = self.decomp2(first_stage_out)
            dec_out = self.dec_embedding(seasonal_second, x_mark_dec[:, -self.pred_len:, :])
            seasonal, trend = self.decoder(dec_out, first_stage_out, x_mask=None,
                                           trend=trend_second[:, -self.pred_len:, :])
            output = trend + seasonal
            if self.output_attention:
                return output[:, -self.pred_len:, :], _
            else:
                return output[:, -self.pred_len:, :]  # [B, L, D]


class GenericBasis(nn.Module):
    """
    Generic basis function.
    """
    def __init__(self, backcast_size, forecast_size):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta):
        return theta[:, :self.backcast_size], theta[:, -self.forecast_size:]


class TrendBasis(nn.Module):
    """
    Polynomial function to model trend.
    """
    def __init__(self, degree_of_polynomial, backcast_size, forecast_size):
        super().__init__()
        self.polynomial_size = degree_of_polynomial + 1  # degree of polynomial with constant term
        self.backcast_time = nn.Parameter(
            torch.tensor(np.concatenate([np.power(np.arange(backcast_size, dtype=np.float) / backcast_size, i)[None, :]
                                     for i in range(self.polynomial_size)]), dtype=torch.float32),
            requires_grad=False)
        self.forecast_time = nn.Parameter(
            torch.tensor(np.concatenate([np.power(np.arange(forecast_size, dtype=np.float) / forecast_size, i)[None, :]
                                     for i in range(self.polynomial_size)]), dtype=torch.float32), requires_grad=False)

    def forward(self, theta):
        backcast = torch.einsum('bp,pt->bt', theta[:, self.polynomial_size:], self.backcast_time)
        forecast = torch.einsum('bp,pt->bt', theta[:, :self.polynomial_size], self.forecast_time)
        return backcast, forecast


class SeasonalityBasis(nn.Module):
    """
    Harmonic functions to model seasonality.
    """
    def __init__(self, harmonics, backcast_size, forecast_size):
        super().__init__()
        self.frequency = np.append(np.zeros(1, dtype=np.float32),
                                   np.arange(harmonics, harmonics / 2 * forecast_size,
                                             dtype=np.float32) / harmonics)[None, :]
        backcast_grid = -2 * np.pi * (
                np.arange(backcast_size, dtype=np.float32)[:, None] / forecast_size) * self.frequency
        forecast_grid = 2 * np.pi * (
                np.arange(forecast_size, dtype=np.float32)[:, None] / forecast_size) * self.frequency
        self.backcast_cos_template = nn.Parameter(torch.tensor(np.transpose(np.cos(backcast_grid)), dtype=torch.float32),
                                                    requires_grad=False)
        self.backcast_sin_template = nn.Parameter(torch.tensor(np.transpose(np.sin(backcast_grid)), dtype=torch.float32),
                                                    requires_grad=False)
        self.forecast_cos_template = nn.Parameter(torch.tensor(np.transpose(np.cos(forecast_grid)), dtype=torch.float32),
                                                    requires_grad=False)
        self.forecast_sin_template = nn.Parameter(torch.tensor(np.transpose(np.sin(forecast_grid)), dtype=torch.float32),
                                                    requires_grad=False)

    def forward(self, theta):
        params_per_harmonic = theta.shape[1] // 4
        backcast_harmonics_cos = torch.einsum('bp,pt->bt', theta[:, 2 * params_per_harmonic:3 * params_per_harmonic],
                                          self.backcast_cos_template)
        backcast_harmonics_sin = torch.einsum('bp,pt->bt', theta[:, 3 * params_per_harmonic:], self.backcast_sin_template)
        backcast = backcast_harmonics_sin + backcast_harmonics_cos
        forecast_harmonics_cos = torch.einsum('bp,pt->bt',
                                          theta[:, :params_per_harmonic], self.forecast_cos_template)
        forecast_harmonics_sin = torch.einsum('bp,pt->bt', theta[:, params_per_harmonic:2 * params_per_harmonic],
                                          self.forecast_sin_template)
        forecast = forecast_harmonics_sin + forecast_harmonics_cos

        return backcast, forecast
