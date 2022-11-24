import torch
import torch.nn as nn
from Self_Regression.decoder import Decoder, DecoderLayer
from Self_Regression.attn import FullAttention, ProbAttention, AttentionLayer
from Self_Regression.embed import DataEmbedding
from GBT.Auto_Regression import AR
from FEDformer.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from FEDformer.Autoformer_EncDec import Auto_Encoder, Auto_EncoderLayer, Auto_Decoder, Auto_DecoderLayer,\
    my_Layernorm, series_decomp, series_decomp_multi
from utils.RevIN import RevIN


class GBT(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=[3, 2, 1], d_layers=2, auto_d_layers=1,
                 dropout=0.0, attn='prob', time=True, activation='gelu',
                 output_attention=False, distil=True, mix=True, feature_extractor='Attention',
                 kernel=3, fd_model=64, moving_avg=[24], instance=False,
                 use_RevIN=False, format='transformer', device=torch.device('cuda:0')):
        super(GBT, self).__init__()
        self.pred_len = out_len
        self.label_len = label_len
        self.attn = attn
        self.output_attention = output_attention
        if instance:
            enc_in = 1
            dec_in = 1
            c_out = 1

        self.group = 1
        self.n_heads = n_heads
        self.format = format

        # Encoding
        print("Start Embedding")
        # decoding
        self.dec_embedding = DataEmbedding(dec_in, d_model, dropout, True, time, group=self.group)
        print("Embedding finished")

        self.use_RevIN = use_RevIN
        if use_RevIN:
            self.revin = RevIN(enc_in)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder
        if format == 'transformer':
            self.AR = AR(enc_in, c_out, label_len, out_len, feature_extractor, kernel=kernel,
                         group=not mix, block_nums=e_layers[0], time=time,
                         fd_model=fd_model, sd_model=d_model, pyramid=len(e_layers), dropout=dropout)
        elif format == 'autoformer':
            self.enc_embedding = DataEmbedding(enc_in, d_model, dropout, position=False, time=time, group=self.group)
            kernel_size = moving_avg
            if isinstance(kernel_size, list):
                self.decomp = series_decomp_multi(kernel_size)
                self.decomp2 = series_decomp_multi(kernel_size)
            else:
                self.decomp = series_decomp(kernel_size)
                self.decomp2 = series_decomp(kernel_size)
            self.encoder = Auto_Encoder(
                [
                    Auto_EncoderLayer(
                        AutoCorrelationLayer(
                            AutoCorrelation(False, factor, attention_dropout=dropout,
                                            output_attention=output_attention),
                            d_model, n_heads),
                        d_model,
                        moving_avg=moving_avg,
                        dropout=dropout,
                        activation=activation,
                        trend=False
                    ) for l in range(e_layers[0])
                ],
                norm_layer=my_Layernorm(d_model)
            )
            self.decoder_auto = Auto_Decoder(
                [
                    Auto_DecoderLayer(
                        AutoCorrelationLayer(
                            AutoCorrelation(True, factor, attention_dropout=dropout,
                                            output_attention=False),
                            d_model, n_heads),
                        AutoCorrelationLayer(
                            AutoCorrelation(False, factor, attention_dropout=dropout,
                                            output_attention=False),
                            d_model, n_heads),
                        d_model,
                        c_out,
                        moving_avg=moving_avg,
                        dropout=dropout,
                        activation=activation,
                    )
                    for l in range(auto_d_layers)
                ],
                norm_layer=my_Layernorm(d_model),
                projection=nn.Linear(d_model, c_out, bias=True)
            )
        else:
            print('format error')
            exit(-1)
        if format == 'transformer':
            # Decoder
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                       d_model, self.n_heads, mix=mix, group=self.group),
                        d_model,
                        dropout=dropout,
                        activation=activation,
                        group=self.group
                    )
                    for l in range(d_layers)
                ],
                norm_layer=torch.nn.LayerNorm(d_model)
            )
            self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, groups=self.group,
                                        bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, flag='first stage'):
        if self.use_RevIN:
            x_dec[:, :self.label_len, :] = self.revin(x_dec[:, :self.label_len, :], 'norm')
        if self.format == 'transformer':
            first_stage_out = self.AR(x_dec[:, :self.label_len, :], x_mark_dec[:, :self.label_len, :], flag)
        elif self.format == 'autoformer':
            # decomp init
            mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
            zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
            seasonal_init, trend_init = self.decomp(x_enc)
            # decoder input
            trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
            seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
            # enc
            enc_out = self.enc_embedding(x_enc, x_mark_enc)
            enc_out, attns = self.encoder(enc_out, attn_mask=None)
            # dec
            dec_out = self.dec_embedding(seasonal_init, x_mark_dec[:, -self.label_len - self.pred_len:, :])
            seasonal_part, trend_part = self.decoder_auto(dec_out, enc_out, x_mask=None, cross_mask=None,
                                                          trend=trend_init)
            # final
            first_stage_out = trend_part + seasonal_part
            first_stage_out = first_stage_out[:, -self.pred_len:, :]
        else:
            exit(-1)
        if self.use_RevIN:
            first_stage_out = self.revin(first_stage_out, 'denorm')
        if flag == 'first stage':
            return first_stage_out
        else:
            first_stage_out = first_stage_out.clone().detach()
        if self.format == 'transformer':
            dec_out = self.dec_embedding(first_stage_out, x_mark_dec[:, -self.pred_len:, :])
            dec_out = self.decoder(dec_out)
            output = self.projection(dec_out.permute(0, 2, 1)).transpose(1, 2)
            if self.output_attention:
                return output[:, -self.pred_len:, :] + first_stage_out[:, -self.pred_len:, :], _
            else:
                return output[:, -self.pred_len:, :] + first_stage_out[:, -self.pred_len:, :]  # [B, L, D]
        elif self.format == 'autoformer':
            # Self-Regression stage
            seasonal_second, trend_second = self.decomp2(first_stage_out)
            dec_out = self.dec_embedding(seasonal_second, x_mark_dec[:, -self.pred_len:, :])
            seasonal, trend = self.decoder_auto(dec_out, enc_out.detach(), x_mask=None,
                                                trend=trend_second[:, -self.pred_len:, :])
            # final
            output = trend + seasonal
            if self.output_attention:
                return output[:, -self.pred_len:, :], _
            else:
                return output[:, -self.pred_len:, :]  # [B, L, D]
