import torch
import torch.nn as nn
import torch.nn.functional as F
from FEDformer.AutoCorrelation import AutoCorrelationLayer
from FEDformer.FourierCorrelation import FourierBlock
from FEDformer.MultiWaveletCorrelation import MultiWaveletTransform
from FEDformer.Autoformer_EncDec import Auto_Encoder, Auto_EncoderLayer, my_Layernorm, series_decomp, series_decomp_multi
from Self_Regression.decoder import Decoder, DecoderLayer
from Self_Regression.attn import AttentionLayer, FullAttention
from Self_Regression.embed import DataEmbedding
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FEDformer(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    """
    def __init__(self, seq_len, label_len, pred_len, version, mode_select, modes,
                 output_attention, moving_avg, enc_in, dec_in, d_model,
                 dropout, time, factor, L, base, c_out, n_heads, activation, e_layers, d_layers, instance,
                 device=torch.device('cuda:0')):
        super(FEDformer, self).__init__()
        if instance:
            dec_in = 1
            enc_in = 1
            c_out = 1
        self.version = version
        self.mode_select = mode_select
        self.modes = modes
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.c_out = c_out
        self.output_attention = output_attention

        # Decomp
        kernel_size = moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        # Embedding
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout, position=False, time=time)
        self.dec_embedding = DataEmbedding(dec_in, d_model, dropout, position=True, time=time)

        if version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(ich=d_model, L=L, base=base)
            decoder_self_att = FullAttention(mask_flag=True, attention_dropout=dropout,
                                             output_attention=output_attention)
        else:
            encoder_self_att = FourierBlock(in_channels=d_model,
                                            out_channels=d_model,
                                            seq_len=self.seq_len,
                                            modes=modes,
                                            mode_select_method=mode_select)
            decoder_self_att = FullAttention(mask_flag=True, attention_dropout=dropout,
                                             output_attention=output_attention)

        self.encoder = Auto_Encoder(
            [
                Auto_EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
                        d_model, n_heads),
                    d_model,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                    projection=nn.Linear(d_model, enc_in, bias=True)
                ) for l in range(e_layers)
            ],
            norm_layer=my_Layernorm(d_model),
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(decoder_self_att, d_model, n_heads, mix=True, group=1),
                    d_model,
                    dropout=dropout,
                    activation=activation,
                    group=1
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.flatten = nn.Flatten()
        self.projection1 = nn.Linear(seq_len * d_model, pred_len * c_out)
        self.projection2 = nn.Linear(in_features=seq_len, out_features=pred_len)
        self.projection3 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, groups=1,
                                     bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, flag='first stage'):
        # decomp init
        seasonal_init, trend_init = self.decomp(x_enc)
        # Auto-Regression stage
        enc_out = self.enc_embedding(seasonal_init, x_mark_enc)
        season_first, trend_first, attns = self.encoder(enc_out, trend_init, attn_mask=enc_self_mask)
        season_first = self.flatten(season_first)
        season_first = self.projection1(season_first).contiguous().view(-1, self.pred_len, self.c_out)
        trend_first = self.projection2(trend_first.permute(0, 2, 1)).transpose(2, 1)
        if flag == 'first stage':
            return season_first + trend_first
        else:
            season_first = season_first.clone().detach()
            trend_first = trend_first.clone().detach()
        # Self-Regression stage
        dec_out = self.dec_embedding(season_first + trend_first, x_mark_dec[:, -self.pred_len:, :])
        dec_out = self.decoder(dec_out, x_mask=dec_self_mask)
        dec_out = self.projection3(dec_out.permute(0, 2, 1)).transpose(1, 2)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :] + season_first + trend_first, attns
        else:
            return dec_out[:, -self.pred_len:, :] + season_first + trend_first  # [B, L, D]
