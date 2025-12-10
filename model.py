import math
from random import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.layers import trunc_normal_
from torch.autograd import Variable

from torch.nn import TransformerEncoderLayer, TransformerEncoder

class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A, dims):
        if dims == 2:
            x
        elif dims == 3:
            x
        else:
            raise NotImplementedError('PGCN not implemented for A of dimension ' + str(dims))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a, a.dim())
            out.append(x1)

            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a, a.dim())
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class Transformer(nn.Module):
    def __init__(
        self, patch_size=12, in_channel=1, out_channel=32, dropout=0.1, mask_size=12, mask_ratio=0.75, num_encoder_layers=4, mode="pretrain"
    ):
        super().__init__()
        self.patch_size = patch_size
        self.selected_feature = 0
        self.mode = mode
        self.patch = InputEmbedding(patch_size, in_channel, out_channel)
        self.pe = PositionalEncoding(out_channel, dropout=dropout)
        self.mask = MaskGenerator(mask_size, mask_ratio)
        self.encoder = TransformerLayers(out_channel, num_encoder_layers)
        self.encoder_2_decoder = nn.Linear(out_channel, out_channel)
        self.decoder = TransformerLayers(out_channel, 1)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, out_channel))
        trunc_normal_(self.mask_token, std=0.02)
        self.output_layer = nn.Linear(out_channel, patch_size)

    def _forward_pretrain(self, input):
        input = input.transpose(1, 2)
        batch_size, num_nodes, num_features, long_seq_len = input.size()
        patches = self.patch(input)
        patches = patches.transpose(-1, -2)
        patches = self.pe(patches)

        indices_not_masked, indices_masked = self.mask()
        repr_not_masked = patches[:, :, indices_not_masked, :]

        hidden_not_masked = self.encoder(repr_not_masked)
        hidden_not_masked = self.encoder_2_decoder(hidden_not_masked)
        hidden_masked = self.pe(
            self.mask_token.expand(batch_size, num_nodes, len(indices_masked), hidden_not_masked.size(-1)),
            indices=indices_masked
        )
        hidden = torch.cat([hidden_not_masked, hidden_masked], dim=-2)
        hidden = self.decoder(hidden)

        output = self.output_layer(hidden)
        output_masked = output[:, :, len(indices_not_masked) :, :]
        output_masked = output_masked.view(batch_size, num_nodes, -1).transpose(1, 2)

        labels = (
        )
        labels_masked = labels[:, :, indices_masked, :].contiguous()
        labels_masked = labels_masked.view(batch_size, num_nodes, -1).transpose(1, 2)
        return output_masked, labels_masked

    def _forward_backend(self, input):
        patches = self.patch(input)
        patches = patches.transpose(-1, -2)
        patches = self.pe(patches)
        hidden = self.encoder(patches)
        return hidden

    def forward(self, input_data):
        if self.mode == "pretrain":
            out = self._forward_pretrain(input_data)
            return out
        else:
            out = self._forward_backend(input_data)
            return out

class InputEmbedding(nn.Module):
    def __init__(self, patch_size, input_channel, output_channel):
        super().__init__()
        self.output_channel = output_channel
        self.patch_size = patch_size
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.input_embedding = nn.Conv2d(
            32, output_channel, kernel_size=(1, 1), stride=(1, 1)
        )

    def forward(self, input):
        input = input.transpose(1, 2)
        batch_size, num_nodes, num_channels, long_seq_len = input.size()
        input = input.unsqueeze(-1)
        input = input.reshape(batch_size * num_nodes, num_channels, long_seq_len, 1)
        output = self.input_embedding(input)
        output = output.squeeze(-1).view(batch_size, num_nodes, self.output_channel, -1)
        return output


class LearnableTemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.empty(max_len, d_model), requires_grad=True)

    def forward(self, input, indices):
        if indices is None:
            pe = self.pe[: input.size(1), :].unsqueeze(0)
        else:
            pe = self.pe[indices].unsqueeze(0)
        x = input + pe
        x = self.dropout(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.tem_pe = LearnableTemporalPositionalEncoding(hidden_dim, dropout)

    def forward(self, input, indices=None):
        batch_size, num_nodes, num_subseq, out_channels = input.size()
        input = self.tem_pe(input.view(batch_size * num_nodes, num_subseq, out_channels), indices=indices)
        input = input.view(batch_size, num_nodes, num_subseq, out_channels)
        return input


class TransformerLayers(nn.Module):
    def __init__(self, hidden_dim, num_layers, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = hidden_dim
        encoder_layers = TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

    def forward(self, input):
        batch_size, num_nodes, num_subseq, out_channels = input.size()
        x = input * math.sqrt(self.d_model)
        x = x.view(batch_size * num_nodes, num_subseq, out_channels)
        x = x.transpose(0, 1)
        output = self.transformer_encoder(x, mask=None)
        output = output.transpose(0, 1).view(batch_size, num_nodes, num_subseq, out_channels)
        return output

class MaskGenerator(nn.Module):
    def __init__(self, mask_size, mask_ratio):
        super().__init__()
        self.mask_size = mask_size
        self.mask_ratio = mask_ratio
        self.sort = True

    def uniform_rand(self):
        mask = list(range(int(self.mask_size)))
        mask_len = int(self.mask_size * self.mask_ratio)
        if self.sort:
            self.masked_tokens = sorted(self.masked_tokens)
            self.unmasked_tokens = sorted(self.unmasked_tokens)
        return self.unmasked_tokens, self.masked_tokens

    def forward(self):
        self.unmasked_tokens, self.masked_tokens = self.uniform_rand()
        return self.unmasked_tokens, self.masked_tokens


class PGSFormer(nn.Module):
    def __init__(self, device, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True,
                 in_dim=2, out_dim=12, residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=512,
                 kernel_size=2, blocks=4, layers=2):
        super(PGSFormer, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if supports is None:
                self.supports = []
            self.adpvec = nn.Parameter(torch.randn(12, 12).to(device), requires_grad=True).to(device)
            self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1, 1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field

        self.transformer = Transformer(mode="inference")
        self.MLP1 = nn.Conv2d(in_channels=64,
                              out_channels=32,
                              kernel_size=(1, 1))
        self.MLP2 = nn.Conv2d(in_channels=32,
                              out_channels=64,
                              kernel_size=(1, 1))

    def forward(self, input):
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:

            xn = input[:, 0, :, -12:]
            xn = (xn - xn.min(dim=-1)[0].unsqueeze(-1)) / \
                 (xn.max(dim=-1)[0] - xn.min(dim=-1)[0]).unsqueeze(-1)
            xn = torch.nan_to_num(xn, nan=0.5)
            adp = torch.einsum('nvt, tc->nvc', (xn, self.adpvec))
            adp = torch.bmm(adp, xn.permute(0, 2, 1))
            adp = F.softmax(F.relu(adp), dim=1)

            new_supports = self.supports + [adp]

        for i in range(self.blocks * self.layers):
            residual = x
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = self.transformer(x)
            x = x.permute(0, 3, 1, 2)
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        return x





