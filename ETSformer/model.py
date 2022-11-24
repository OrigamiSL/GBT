import torch
import torch.nn as nn

from ETSformer.modules import ETSEmbedding
from ETSformer.encoder import EncoderLayer, Encoder
from ETSformer.decoder import ETSDecoderLayer, ETSDecoder
from Self_Regression.decoder import DecoderLayer, Decoder
from Self_Regression.attn import AttentionLayer, FullAttention
from Self_Regression.embed import DataEmbedding


class Transform:
    def __init__(self, sigma):
        self.sigma = sigma

    @torch.no_grad()
    def transform(self, x):
        return self.jitter(self.shift(self.scale(x)))

    def jitter(self, x):
        return x + (torch.randn(x.shape).to(x.device) * self.sigma)

    def scale(self, x):
        return x * (torch.randn(x.size(-1)).to(x.device) * self.sigma + 1)

    def shift(self, x):
        return x + (torch.randn(x.size(-1)).to(x.device) * self.sigma)


class ETSformer(nn.Module):

    def __init__(self, seq_len, label_len, pred_len, enc_in, dec_in, d_model, dropout,
                 time, K, sigma, c_out, n_heads, activation, e_layers, d_layers, output_attention, instance,
                 device=torch.device('cuda:0')):
        super().__init__()
        if instance:
            enc_in = 1
            dec_in = 1
            c_out = 1
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.output_attention = output_attention

        # assert d_layers == e_layers

        # Embedding
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout, position=False, time=time)
        self.dec_embedding = DataEmbedding(dec_in, d_model, dropout, position=True, time=time)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    d_model, n_heads, c_out, pred_len, K,
                    dropout=dropout,
                    activation=activation,
                ) for _ in range(e_layers)
            ]
        )
        self.ETSdecoder = ETSDecoder(
            [
                ETSDecoderLayer(
                    d_model, n_heads, c_out, pred_len,
                    dropout=dropout,
                ) for _ in range(e_layers)
            ],
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(FullAttention(True, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, group=1),
                    d_model,
                    dropout=dropout,
                    activation=activation,
                    group=1
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.transform = Transform(sigma=sigma)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, groups=1,
                                    bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, flag='first stage'):
        # with torch.no_grad():
        #     if self.training:
        #         x_enc = self.transform.transform(x_enc)
        res = self.enc_embedding(x_enc, x_mark_enc)
        level, growths, seasons = self.encoder(res, x_enc, attn_mask=None)
        growth, season = self.ETSdecoder(growths, seasons)
        first_stage_out = level[:, -1:] + growth + season
        if flag == 'first stage':
            return first_stage_out
        else:
            first_stage_out = first_stage_out.clone().detach()
            dec_out = self.dec_embedding(first_stage_out, x_mark_dec[:, self.label_len:, :])
            dec_out = self.decoder(dec_out)
            dec_out = self.projection(dec_out.permute(0, 2, 1)).transpose(1, 2)

            if self.output_attention:
                return dec_out[:, -self.pred_len:, :] + first_stage_out, _
            else:
                return dec_out[:, -self.pred_len:, :] + first_stage_out  # [B, L, D]
