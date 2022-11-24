import torch
import torch.nn as nn

from GBT.ConvBlock import ConvLayer, ConvBlock
from Self_Regression.embed import DataEmbedding


class Encoder(nn.Module):
    def __init__(self, d_model, kernel, dropout, group, block_nums, label_len, pred_len, c_out, FE='ResNet'):
        super(Encoder, self).__init__()
        pro_conv = [ConvBlock(d_model * (2 ** i), d_model * (2 ** (i + 1)),
                              kernel=kernel, dropout=dropout, group=group, FE=FE)
                    for i in range(block_nums)]
        self.pro_conv = nn.ModuleList(pro_conv)
        last_dim = d_model * (2 ** block_nums)
        self.F = nn.Flatten()
        self.projection = nn.Conv1d(in_channels=last_dim * label_len // (2 ** block_nums),
                                    out_channels=pred_len * c_out, kernel_size=1, groups=group)
        self.c_out = c_out
        self.pred_len = pred_len

    def forward(self, x):
        for conv in self.pro_conv:
            x = conv(x)
        F_out = self.F(x.permute(0, 2, 1)).unsqueeze(-1)
        x_out = self.projection(F_out).squeeze().contiguous().view(-1, self.c_out, self.pred_len)
        x_out = x_out.transpose(1, 2)
        return x_out


class FC_block(nn.Module):
    def __init__(self, in_channels, out_channels, group):
        super(FC_block, self).__init__()
        self.FC = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, groups=group)

    def forward(self, x):
        return self.FC(x)


class AR(nn.Module):
    def __init__(self, enc_in, c_out, label_len, pred_len, feature_extractor, kernel=3,
                 group=False, block_nums=3, time=False,
                 fd_model=64, sd_model=512, pyramid=1, dropout=0.0):
        super(AR, self).__init__()
        print("Start Embedding")
        # Enbeddinging
        self.group = 1
        self.pyramid = pyramid

        self.time = time
        self.enc_bed = [DataEmbedding(enc_in, fd_model, dropout, position=False, time=self.time, group=self.group)
                        for i in range(pyramid)]
        self.enc_bed = nn.ModuleList(self.enc_bed)

        assert (pyramid <= block_nums)
        self.label_len = label_len
        self.pred_len = pred_len
        self.c_out = c_out
        self.fd_model = fd_model
        self.sd_model = sd_model
        print("Embedding finished")

        Encoders = [Encoder(fd_model, kernel, dropout, self.group, block_nums - i,
                            label_len // (2 ** i), pred_len, c_out, FE=feature_extractor)
                    for i in range(pyramid)]
        self.Encoders = nn.ModuleList(Encoders)

    def forward(self, x_enc, x_mark, flag='first stage'):
        enc_input = x_enc
        i = 0
        enc_out = 0
        for embed, RT_b in zip(self.enc_bed, self.Encoders):
            embed_enc = embed(enc_input[:, -self.label_len // (2 ** i):, :], x_mark[:, -self.label_len // (2 ** i):, :])
            enc_out += RT_b(embed_enc)
            i += 1
        enc_out = enc_out / i
        if flag == 'first stage':
            return enc_out  # [B, L, D]
        elif flag == 'second stage':
            return enc_out.clone().detach()  # [B, L, D]
        else:
            print('invalid stages')
            exit(-1)
