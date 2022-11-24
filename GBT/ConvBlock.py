import torch
import torch.nn as nn
import math
from torch.nn.utils import weight_norm


class ConvLayer(nn.Module):
    def __init__(self, c_in, c_out, kernel=3, dropout=0, group=1, s=1):
        super(ConvLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.kernel = kernel
        self.downConv = weight_norm(nn.Conv1d(in_channels=c_in,
                                              out_channels=c_in,
                                              padding=(kernel - 1) // 2,
                                              stride=s,
                                              groups=group,
                                              kernel_size=kernel))
        self.activation1 = nn.GELU()
        self.actConv = weight_norm(nn.Conv1d(in_channels=c_in,
                                             out_channels=c_out,
                                             padding=1,
                                             stride=1,
                                             groups=group,
                                             kernel_size=3))
        self.activation2 = nn.GELU()
        self.sampleConv = nn.Conv1d(in_channels=c_in,
                                    out_channels=c_out,
                                    groups=group,
                                    kernel_size=1) if c_in != c_out else None
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1) if s != 1 else None

    def forward(self, x):
        y = x.clone()
        if self.sampleConv is not None:
            y = self.sampleConv(y.permute(0, 2, 1)).transpose(1, 2)
        if self.pool is not None:
            y = self.pool(y.permute(0, 2, 1)).transpose(1, 2)
        x = self.dropout(self.downConv(x.permute(0, 2, 1)))
        x = self.activation1(x).transpose(1, 2)
        x = self.dropout(self.actConv(x.permute(0, 2, 1)))
        x = self.activation2(x).transpose(1, 2)
        x = x + y
        return x


class CspConvLayer(nn.Module):
    def __init__(self, c_in, c_out, kernel=3, dropout=0, group=1, s=1):
        super(CspConvLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.kernel = kernel
        self.downConv = weight_norm(nn.Conv1d(in_channels=c_in // 2,
                                              out_channels=c_in // 2,
                                              padding=(kernel - 1) // 2,
                                              stride=s,
                                              groups=group,
                                              kernel_size=kernel))
        self.activation1 = nn.GELU()
        self.actConv = weight_norm(nn.Conv1d(in_channels=c_in // 2,
                                             out_channels=c_out // 2,
                                             padding=1,
                                             stride=1,
                                             groups=group,
                                             kernel_size=3))
        self.activation2 = nn.GELU()
        self.sampleConv = nn.Conv1d(in_channels=c_in // 2,
                                    out_channels=c_out // 2,
                                    groups=group,
                                    kernel_size=1) if c_in != c_out else None
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1) if s != 1 else None

        self.csp_sampleConv = nn.Conv1d(in_channels=c_in // 2,
                                        out_channels=c_out // 2,
                                        groups=group,
                                        kernel_size=1) if c_in != c_out else None
        self.csp_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1) if s != 1 else None

    def forward(self, x):
        y = x[:, :, 0::2].clone()
        csp_x = x[:, :, 1::2].clone()
        if self.sampleConv is not None:
            y = self.sampleConv(y.permute(0, 2, 1)).transpose(1, 2)
            csp_x = self.csp_sampleConv(csp_x.permute(0, 2, 1)).transpose(1, 2)
        if self.pool is not None:
            y = self.pool(y.permute(0, 2, 1)).transpose(1, 2)
            csp_x = self.pool(csp_x.permute(0, 2, 1)).transpose(1, 2)
        x = self.dropout(self.downConv(x[:, :, 0::2].permute(0, 2, 1)))
        x = self.activation1(x).transpose(1, 2)
        x = self.dropout(self.actConv(x.permute(0, 2, 1)))
        x = self.activation2(x).transpose(1, 2)
        x = x + y

        B, L, D = x.shape
        concat_x = torch.zeros(B, L, 2 * D).to(x.device)
        concat_x[:, :, 0::2] = x
        concat_x[:, :, 1::2] = csp_x

        return concat_x


class AttentionBlock(nn.Module):
    def __init__(self, dropout=0.1):
        super(AttentionBlock, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, d_model, dropout=0, group=1):
        super(AttentionLayer, self).__init__()

        self.inner_attention = AttentionBlock(dropout)
        self.query_projection = weight_norm(nn.Conv1d(in_channels=d_model,
                                                      out_channels=d_model,
                                                      groups=group,
                                                      kernel_size=1))
        self.key_projection = weight_norm(nn.Conv1d(in_channels=d_model,
                                                    out_channels=d_model,
                                                    groups=group,
                                                    kernel_size=1))
        self.value_projection = weight_norm(nn.Conv1d(in_channels=d_model,
                                                      out_channels=d_model,
                                                      groups=group,
                                                      kernel_size=1))
        self.out_projection = weight_norm(nn.Conv1d(in_channels=d_model,
                                                    out_channels=d_model,
                                                    groups=group,
                                                    kernel_size=1))
        self.n_heads = 8

    def forward(self, queries):
        B, L, _ = queries.shape
        keys = queries.clone()
        values = queries.clone()
        _, S, _ = keys.shape
        H = self.n_heads

        intial_queries = queries.clone()

        queries = self.query_projection(queries.permute(0, 2, 1)).transpose(1, 2).view(B, L, H, -1)
        keys = self.key_projection(keys.permute(0, 2, 1)).transpose(1, 2).view(B, S, H, -1)
        values = self.value_projection(values.permute(0, 2, 1)).transpose(1, 2).view(B, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values
        )

        out = self.out_projection(out.view(B, L, -1).permute(0, 2, 1)).transpose(1, 2)

        return intial_queries + out


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel=3, dropout=0, pool=True, group=1, FE='ResNet'):
        super(ConvBlock, self).__init__()
        if FE != 'Attention':
            FE_block = ConvLayer if FE == 'ResNet' else CspConvLayer
            if pool:
                self.conv = nn.Sequential(
                    FE_block(c_in, c_in, kernel, dropout, group=group, s=2),
                    FE_block(c_in, c_out, kernel, dropout, group=group, s=1)
                )
            else:
                self.conv = nn.Sequential(
                    FE_block(c_in, c_in, kernel, dropout, group=group, s=1),
                    FE_block(c_in, c_out, kernel, dropout, group=group, s=1)
                )
        else:
            assert (pool is True)
            self.conv = nn.Sequential(
                AttentionLayer(c_in, dropout, group=group),
                ConvLayer(c_in, c_out, kernel, dropout, group=group, s=2)
            )

    def forward(self, x):
        x = self.conv(x)
        return x
