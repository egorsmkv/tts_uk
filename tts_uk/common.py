# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# 1x1InvertibleConv and WN based on implementation from WaveGlow https://github.com/NVIDIA/waveglow/blob/master/glow.py
# Original license:
# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import ast
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .partialconv1d import PartialConv1d as pconv1d
from .splines import (
    piecewise_linear_inverse_transform,
    piecewise_linear_transform,
    unbounded_piecewise_quadratic_transform,
)
from .torch_env import device


def get_mask_from_lengths(lengths):
    """Constructs binary mask from a 1D torch tensor of input lengths

    Args:
        lengths (torch.tensor): 1D tensor
    Returns:
        mask (torch.tensor): num_sequences x max_length x 1 binary tensor
    """

    max_len = torch.max(lengths).item()

    ids = torch.tensor(list(range(max_len)), dtype=torch.long, device=device)

    mask = (ids < lengths.unsqueeze(1)).bool()

    return mask


class ExponentialClass(torch.nn.Module):
    def __init__(self):
        super(ExponentialClass, self).__init__()

    def forward(self, x):
        return torch.exp(x)


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain="linear"):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
        use_partial_padding=False,
        use_weight_norm=False,
    ):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.use_partial_padding = use_partial_padding
        self.use_weight_norm = use_weight_norm
        conv_fn = torch.nn.Conv1d
        if self.use_partial_padding:
            conv_fn = pconv1d
        self.conv = conv_fn(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )
        if self.use_weight_norm:
            self.conv = torch.nn.utils.parametrizations.weight_norm(self.conv)

    def forward(self, signal, mask=None):
        if self.use_partial_padding:
            conv_signal = self.conv(signal, mask)
        else:
            conv_signal = self.conv(signal)
        if mask is not None:
            # always re-zero output if mask is
            # available to match zero-padding
            conv_signal = conv_signal * mask
        return conv_signal


class DenseLayer(nn.Module):
    def __init__(self, in_dim=1024, sizes=[1024, 1024]):
        super(DenseLayer, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [
                LinearNorm(in_size, out_size, bias=True)
                for (in_size, out_size) in zip(in_sizes, sizes)
            ]
        )

    def forward(self, x):
        for linear in self.layers:
            x = torch.tanh(linear(x))
        return x


class LengthRegulator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dur):
        output = []
        for x_i, dur_i in zip(x, dur):
            expanded = self.expand(x_i, dur_i)
            output.append(expanded)
        output = self.pad(output)
        return output

    def expand(self, x, dur):
        output = []
        for i, frame in enumerate(x):
            expanded_len = int(dur[i] + 0.5)
            expanded = frame.expand(expanded_len, -1)
            output.append(expanded)
        output = torch.cat(output, 0)
        return output

    def pad(self, x):
        output = []
        max_len = max([x[i].size(0) for i in range(len(x))])
        for i, seq in enumerate(x):
            padded = F.pad(seq, [0, 0, 0, max_len - seq.size(0)], "constant", 0.0)
            output.append(padded)
        output = torch.stack(output)
        return output


class ConvLSTMLinear(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        n_layers=2,
        n_channels=256,
        kernel_size=3,
        p_dropout=0.1,
        lstm_type="bilstm",
        use_linear=True,
    ):
        super(ConvLSTMLinear, self).__init__()
        self.out_dim = out_dim
        self.lstm_type = lstm_type
        self.use_linear = use_linear
        self.dropout = nn.Dropout(p=p_dropout)

        convolutions = []
        for i in range(n_layers):
            conv_layer = ConvNorm(
                in_dim if i == 0 else n_channels,
                n_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=int((kernel_size - 1) / 2),
                dilation=1,
                w_init_gain="relu",
            )
            conv_layer = torch.nn.utils.parametrizations.weight_norm(
                conv_layer.conv, name="weight"
            )
            convolutions.append(conv_layer)

        self.convolutions = nn.ModuleList(convolutions)

        if not self.use_linear:
            n_channels = out_dim

        if self.lstm_type != "":
            use_bilstm = False
            lstm_channels = n_channels
            if self.lstm_type == "bilstm":
                use_bilstm = True
                lstm_channels = int(n_channels // 2)

            self.bilstm = nn.LSTM(
                n_channels, lstm_channels, 1, batch_first=True, bidirectional=use_bilstm
            )
            lstm_norm_fn_pntr = torch.nn.utils.parametrizations.spectral_norm
            self.bilstm = lstm_norm_fn_pntr(self.bilstm, "weight_hh_l0")
            if self.lstm_type == "bilstm":
                self.bilstm = lstm_norm_fn_pntr(self.bilstm, "weight_hh_l0_reverse")

        if self.use_linear:
            self.dense = nn.Linear(n_channels, out_dim)

    def run_padded_sequence(self, context, lens):
        context_embedded = []
        for b_ind in range(context.size()[0]):  # TODO: speed up
            curr_context = context[b_ind : b_ind + 1, :, : lens[b_ind]].clone()
            for conv in self.convolutions:
                curr_context = self.dropout(F.relu(conv(curr_context)))
            context_embedded.append(curr_context[0].transpose(0, 1))
        context = torch.nn.utils.rnn.pad_sequence(context_embedded, batch_first=True)
        return context

    def run_unsorted_inputs(self, fn, context, lens):
        lens_sorted, ids_sorted = torch.sort(lens, descending=True)
        unsort_ids = [0] * lens.size(0)
        for i in range(len(ids_sorted)):
            unsort_ids[ids_sorted[i]] = i
        lens_sorted = lens_sorted.long().cpu()

        context = context[ids_sorted]
        context = nn.utils.rnn.pack_padded_sequence(
            context, lens_sorted, batch_first=True
        )
        context = fn(context)[0]
        context = nn.utils.rnn.pad_packed_sequence(context, batch_first=True)[0]

        # map back to original indices
        context = context[unsort_ids]
        return context

    def forward(self, context, lens):
        if context.size()[0] > 1:
            context = self.run_padded_sequence(context, lens)
            # to B, D, T
            context = context.transpose(1, 2)
        else:
            for conv in self.convolutions:
                context = self.dropout(F.relu(conv(context)))

        if self.lstm_type != "":
            context = context.transpose(1, 2)
            self.bilstm.flatten_parameters()
            if lens is not None:
                context = self.run_unsorted_inputs(self.bilstm, context, lens)
            else:
                context = self.bilstm(context)[0]
            context = context.transpose(1, 2)

        x_hat = context
        if self.use_linear:
            x_hat = self.dense(context.transpose(1, 2)).transpose(1, 2)

        return x_hat

    def infer(self, z, txt_enc, spk_emb):
        x_hat = self.forward(txt_enc, spk_emb)["x_hat"]
        x_hat = self.feature_processing.denormalize(x_hat)
        return x_hat


class Encoder(nn.Module):
    """Encoder module:
    - Three 1-d convolution banks
    - Bidirectional LSTM
    """

    def __init__(
        self,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,
        encoder_kernel_size=5,
        norm_fn=nn.BatchNorm1d,
        lstm_norm_fn=None,
    ):
        super(Encoder, self).__init__()

        convolutions = []
        for _ in range(encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(
                    encoder_embedding_dim,
                    encoder_embedding_dim,
                    kernel_size=encoder_kernel_size,
                    stride=1,
                    padding=int((encoder_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="relu",
                    use_partial_padding=True,
                ),
                norm_fn(encoder_embedding_dim, affine=True),
            )
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(
            encoder_embedding_dim,
            int(encoder_embedding_dim / 2),
            1,
            batch_first=True,
            bidirectional=True,
        )
        if lstm_norm_fn is not None:
            if "spectral" in lstm_norm_fn:
                print("Applying spectral norm to text encoder LSTM")
                lstm_norm_fn_pntr = torch.nn.utils.parametrizations.spectral_norm
            elif "weight" in lstm_norm_fn:
                print("Applying weight norm to text encoder LSTM")
                lstm_norm_fn_pntr = torch.nn.utils.parametrizations.weight_norm
            self.lstm = lstm_norm_fn_pntr(self.lstm, "weight_hh_l0")
            self.lstm = lstm_norm_fn_pntr(self.lstm, "weight_hh_l0_reverse")

    @torch.autocast(device, enabled=False)
    def forward(self, x, in_lens):
        """
        Args:
            x (torch.tensor): N x C x L padded input of text embeddings
            in_lens (torch.tensor): 1D tensor of sequence lengths
        """
        if x.size()[0] > 1:
            x_embedded = []
            for b_ind in range(x.size()[0]):  # TODO: improve speed
                curr_x = x[b_ind : b_ind + 1, :, : in_lens[b_ind]].clone()
                for conv in self.convolutions:
                    curr_x = F.dropout(F.relu(conv(curr_x)), 0.5, self.training)
                x_embedded.append(curr_x[0].transpose(0, 1))
            x = torch.nn.utils.rnn.pad_sequence(x_embedded, batch_first=True)
        else:
            for conv in self.convolutions:
                x = F.dropout(F.relu(conv(x)), 0.5, self.training)
            x = x.transpose(1, 2)

        # recent amp change -- change in_lens to int
        in_lens = in_lens.int().cpu()

        x = nn.utils.rnn.pack_padded_sequence(x, in_lens, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        return outputs

    @torch.autocast(device, enabled=False)
    def infer(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs


class Invertible1x1ConvLUS(torch.nn.Module):
    def __init__(self, c, cache_inverse=False):
        super(Invertible1x1ConvLUS, self).__init__()
        # Sample a random orthonormal matrix to initialize weights
        W = torch.linalg.qr(torch.FloatTensor(c, c).normal_())[0]
        # Ensure determinant is 1.0 not -1.0
        if torch.det(W) < 0:
            W[:, 0] = -1 * W[:, 0]
        p, lower, upper = torch.lu_unpack(*torch.linalg.lu_factor(W))

        self.register_buffer("p", p)
        # diagonals of lower will always be 1s anyway
        lower = torch.tril(lower, -1)
        lower_diag = torch.diag(torch.eye(c, c))
        self.register_buffer("lower_diag", lower_diag)
        self.lower = nn.Parameter(lower)
        self.upper_diag = nn.Parameter(torch.diag(upper))
        self.upper = nn.Parameter(torch.triu(upper, 1))
        self.cache_inverse = cache_inverse

    @torch.autocast(device, enabled=False)
    def forward(self, z, inverse=False):
        U = torch.triu(self.upper, 1) + torch.diag(self.upper_diag)
        L = torch.tril(self.lower, -1) + torch.diag(self.lower_diag)
        W = torch.mm(self.p, torch.mm(L, U))
        if inverse:
            if not hasattr(self, "W_inverse"):
                # inverse computation
                W_inverse = W.float().inverse()
                if z.is_cuda and z.dtype == torch.float16:
                    W_inverse = W_inverse.half()

                self.W_inverse = W_inverse[..., None]
            z = F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)
            if not self.cache_inverse:
                delattr(self, "W_inverse")
            return z
        else:
            W = W[..., None]
            z = F.conv1d(z, W, bias=None, stride=1, padding=0)
            log_det_W = torch.sum(torch.log(torch.abs(self.upper_diag)))
            return z, log_det_W


class Invertible1x1Conv(torch.nn.Module):
    """
    The layer outputs both the convolution, and the log determinant
    of its weight matrix.  If inverse=True it does convolution with
    inverse
    """

    def __init__(self, c, cache_inverse=False):
        super(Invertible1x1Conv, self).__init__()
        self.conv = torch.nn.Conv1d(
            c, c, kernel_size=1, stride=1, padding=0, bias=False
        )

        # Sample a random orthonormal matrix to initialize weights
        W = torch.qr(torch.FloatTensor(c, c).normal_())[0]

        # Ensure determinant is 1.0 not -1.0
        if torch.det(W) < 0:
            W[:, 0] = -1 * W[:, 0]
        W = W.view(c, c, 1)
        self.conv.weight.data = W
        self.cache_inverse = cache_inverse

    def forward(self, z, inverse=False):
        # DO NOT apply n_of_groups, as it doesn't account for padded sequences
        W = self.conv.weight.squeeze()

        if inverse:
            if not hasattr(self, "W_inverse"):
                # Inverse computation
                W_inverse = W.float().inverse()
                if z.is_cuda and z.dtype == torch.float16:
                    W_inverse = W_inverse.half()

                self.W_inverse = W_inverse[..., None]
            z = F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)
            if not self.cache_inverse:
                delattr(self, "W_inverse")
            return z
        else:
            # Forward computation
            log_det_W = torch.logdet(W).clone()
            z = self.conv(z)
            return z, log_det_W


class SimpleConvNet(torch.nn.Module):
    def __init__(
        self,
        n_mel_channels,
        n_context_dim,
        final_out_channels,
        n_layers=2,
        kernel_size=5,
        with_dilation=True,
        max_channels=1024,
        zero_init=True,
        use_partial_padding=True,
    ):
        super(SimpleConvNet, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.n_layers = n_layers
        in_channels = n_mel_channels + n_context_dim
        out_channels = -1
        self.use_partial_padding = use_partial_padding
        for i in range(n_layers):
            dilation = 2**i if with_dilation else 1
            padding = int((kernel_size * dilation - dilation) / 2)
            out_channels = min(max_channels, in_channels * 2)
            self.layers.append(
                ConvNorm(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=True,
                    w_init_gain="relu",
                    use_partial_padding=use_partial_padding,
                )
            )
            in_channels = out_channels

        self.last_layer = torch.nn.Conv1d(
            out_channels, final_out_channels, kernel_size=1
        )

        if zero_init:
            self.last_layer.weight.data *= 0
            self.last_layer.bias.data *= 0

    def forward(self, z_w_context, seq_lens: torch.Tensor = None):
        # seq_lens: tensor array of sequence sequence lengths
        # output should be b x n_mel_channels x z_w_context.shape(2)
        mask = None
        if seq_lens is not None:
            mask = get_mask_from_lengths(seq_lens).unsqueeze(1).float()

        for i in range(self.n_layers):
            z_w_context = self.layers[i](z_w_context, mask)
            z_w_context = torch.relu(z_w_context)

        z_w_context = self.last_layer(z_w_context)
        return z_w_context


class WN(torch.nn.Module):
    """
    Adapted from WN() module in WaveGlow with modififcations to variable names
    """

    def __init__(
        self,
        n_in_channels,
        n_context_dim,
        n_layers,
        n_channels,
        kernel_size=5,
        affine_activation="softplus",
        use_partial_padding=True,
    ):
        super(WN, self).__init__()
        assert kernel_size % 2 == 1
        assert n_channels % 2 == 0
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        start = torch.nn.Conv1d(n_in_channels + n_context_dim, n_channels, 1)
        start = torch.nn.utils.parametrizations.weight_norm(start, name="weight")
        self.start = start
        self.softplus = torch.nn.Softplus()
        self.affine_activation = affine_activation
        self.use_partial_padding = use_partial_padding
        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        end = torch.nn.Conv1d(n_channels, 2 * n_in_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end

        for i in range(n_layers):
            dilation = 2**i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = ConvNorm(
                n_channels,
                n_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
                use_partial_padding=use_partial_padding,
                use_weight_norm=True,
            )
            # in_layer = nn.Conv1d(n_channels, n_channels, kernel_size,
            #                      dilation=dilation, padding=padding)
            # in_layer = nn.utils.weight_norm(in_layer)
            self.in_layers.append(in_layer)
            res_skip_layer = nn.Conv1d(n_channels, n_channels, 1)
            res_skip_layer = torch.nn.utils.parametrizations.weight_norm(res_skip_layer)
            self.res_skip_layers.append(res_skip_layer)

    def forward(
        self,
        forward_input: Tuple[torch.Tensor, torch.Tensor],
        seq_lens: torch.Tensor = None,
    ):
        z, context = forward_input
        z = torch.cat((z, context), 1)  # append context to z as well
        z = self.start(z)
        output = torch.zeros_like(z)
        mask = None
        if seq_lens is not None:
            mask = get_mask_from_lengths(seq_lens).unsqueeze(1).float()
        non_linearity = torch.relu
        if self.affine_activation == "softplus":
            non_linearity = self.softplus

        for i in range(self.n_layers):
            z = non_linearity(self.in_layers[i](z, mask))
            res_skip_acts = non_linearity(self.res_skip_layers[i](z))
            output = output + res_skip_acts

        output = self.end(output)  # [B, dim, seq_len]
        return output


# Affine Coupling Layers
class SplineTransformationLayerAR(torch.nn.Module):
    def __init__(
        self,
        n_in_channels,
        n_context_dim,
        n_layers,
        affine_model="simple_conv",
        kernel_size=1,
        scaling_fn="exp",
        affine_activation="softplus",
        n_channels=1024,
        n_bins=8,
        left=-6,
        right=6,
        bottom=-6,
        top=6,
        use_quadratic=False,
    ):
        super(SplineTransformationLayerAR, self).__init__()
        self.n_in_channels = n_in_channels  # input dimensions
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
        self.n_bins = n_bins
        self.spline_fn = piecewise_linear_transform
        self.inv_spline_fn = piecewise_linear_inverse_transform
        self.use_quadratic = use_quadratic

        if self.use_quadratic:
            self.spline_fn = unbounded_piecewise_quadratic_transform
            self.inv_spline_fn = unbounded_piecewise_quadratic_transform
            self.n_bins = 2 * self.n_bins + 1
        final_out_channels = self.n_in_channels * self.n_bins

        # autoregressive flow, kernel size 1 and no dilation
        self.param_predictor = SimpleConvNet(
            n_context_dim,
            0,
            final_out_channels,
            n_layers,
            with_dilation=False,
            kernel_size=1,
            zero_init=True,
            use_partial_padding=False,
        )

        # output is unnormalized bin weights

    def normalize(self, z, inverse):
        # normalize to [0, 1]
        if inverse:
            z = (z - self.bottom) / (self.top - self.bottom)
        else:
            z = (z - self.left) / (self.right - self.left)

        return z

    def denormalize(self, z, inverse):
        if inverse:
            z = z * (self.right - self.left) + self.left
        else:
            z = z * (self.top - self.bottom) + self.bottom

        return z

    def forward(self, z, context, inverse=False):
        b_s, c_s, t_s = z.size(0), z.size(1), z.size(2)

        z = self.normalize(z, inverse)

        if z.min() < 0.0 or z.max() > 1.0:
            print("spline z scaled beyond [0, 1]", z.min(), z.max())

        z_reshaped = z.permute(0, 2, 1).reshape(b_s * t_s, -1)
        affine_params = self.param_predictor(context)
        q_tilde = affine_params.permute(0, 2, 1).reshape(b_s * t_s, c_s, -1)
        with torch.autocast(device, enabled=False):
            if self.use_quadratic:
                w = q_tilde[:, :, : self.n_bins // 2]
                v = q_tilde[:, :, self.n_bins // 2 :]
                z_tformed, log_s = self.spline_fn(
                    z_reshaped.float(), w.float(), v.float(), inverse=inverse
                )
            else:
                z_tformed, log_s = self.spline_fn(z_reshaped.float(), q_tilde.float())

        z = z_tformed.reshape(b_s, t_s, -1).permute(0, 2, 1)
        z = self.denormalize(z, inverse)
        if inverse:
            return z

        log_s = log_s.reshape(b_s, t_s, -1)
        log_s = log_s.permute(0, 2, 1)
        log_s = log_s + c_s * (
            np.log(self.top - self.bottom) - np.log(self.right - self.left)
        )
        return z, log_s


class SplineTransformationLayer(torch.nn.Module):
    def __init__(
        self,
        n_mel_channels,
        n_context_dim,
        n_layers,
        with_dilation=True,
        kernel_size=5,
        scaling_fn="exp",
        affine_activation="softplus",
        n_channels=1024,
        n_bins=8,
        left=-4,
        right=4,
        bottom=-4,
        top=4,
        use_quadratic=False,
    ):
        super(SplineTransformationLayer, self).__init__()
        self.n_mel_channels = n_mel_channels  # input dimensions
        self.half_mel_channels = int(n_mel_channels / 2)  # half, because we split
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
        self.n_bins = n_bins
        self.spline_fn = piecewise_linear_transform
        self.inv_spline_fn = piecewise_linear_inverse_transform
        self.use_quadratic = use_quadratic

        if self.use_quadratic:
            self.spline_fn = unbounded_piecewise_quadratic_transform
            self.inv_spline_fn = unbounded_piecewise_quadratic_transform
            self.n_bins = 2 * self.n_bins + 1
        final_out_channels = self.half_mel_channels * self.n_bins

        self.param_predictor = SimpleConvNet(
            self.half_mel_channels,
            n_context_dim,
            final_out_channels,
            n_layers,
            with_dilation=with_dilation,
            kernel_size=kernel_size,
            zero_init=False,
        )

        # output is unnormalized bin weights

    def forward(self, z, context, inverse=False, seq_lens=None):
        b_s, _, t_s = z.size(0), z.size(1), z.size(2)

        # condition on z_0, transform z_1
        n_half = self.half_mel_channels
        z_0, z_1 = z[:, :n_half], z[:, n_half:]

        # normalize to [0,1]
        if inverse:
            z_1 = (z_1 - self.bottom) / (self.top - self.bottom)
        else:
            z_1 = (z_1 - self.left) / (self.right - self.left)

        z_w_context = torch.cat((z_0, context), 1)
        affine_params = self.param_predictor(z_w_context, seq_lens)
        z_1_reshaped = z_1.permute(0, 2, 1).reshape(b_s * t_s, -1)
        q_tilde = affine_params.permute(0, 2, 1).reshape(b_s * t_s, n_half, self.n_bins)

        with torch.autocast(device, enabled=False):
            if self.use_quadratic:
                w = q_tilde[:, :, : self.n_bins // 2]
                v = q_tilde[:, :, self.n_bins // 2 :]
                z_1_tformed, log_s = self.spline_fn(
                    z_1_reshaped.float(), w.float(), v.float(), inverse=inverse
                )
                if not inverse:
                    log_s = torch.sum(log_s, 1)
            else:
                if inverse:
                    z_1_tformed, _dc = self.inv_spline_fn(
                        z_1_reshaped.float(), q_tilde.float(), False
                    )
                else:
                    z_1_tformed, log_s = self.spline_fn(
                        z_1_reshaped.float(), q_tilde.float()
                    )

        z_1 = z_1_tformed.reshape(b_s, t_s, -1).permute(0, 2, 1)

        # undo [0, 1] normalization
        if inverse:
            z_1 = z_1 * (self.right - self.left) + self.left
            z = torch.cat((z_0, z_1), dim=1)
            return z
        else:  # training
            z_1 = z_1 * (self.top - self.bottom) + self.bottom
            z = torch.cat((z_0, z_1), dim=1)
            log_s = log_s.reshape(b_s, t_s).unsqueeze(1) + n_half * (
                np.log(self.top - self.bottom) - np.log(self.right - self.left)
            )
            return z, log_s


class AffineTransformationLayer(torch.nn.Module):
    def __init__(
        self,
        n_mel_channels,
        n_context_dim,
        n_layers,
        affine_model="simple_conv",
        with_dilation=True,
        kernel_size=5,
        scaling_fn="exp",
        affine_activation="softplus",
        n_channels=1024,
        use_partial_padding=False,
    ):
        super(AffineTransformationLayer, self).__init__()
        if affine_model not in ("wavenet", "simple_conv"):
            raise Exception("{} affine model not supported".format(affine_model))
        if isinstance(scaling_fn, list):
            if not all(
                [x in ("translate", "exp", "tanh", "sigmoid") for x in scaling_fn]
            ):
                raise Exception("{} scaling fn not supported".format(scaling_fn))
        else:
            if scaling_fn not in ("translate", "exp", "tanh", "sigmoid"):
                raise Exception("{} scaling fn not supported".format(scaling_fn))

        self.affine_model = affine_model
        self.scaling_fn = scaling_fn
        if affine_model == "wavenet":
            self.affine_param_predictor = WN(
                int(n_mel_channels / 2),
                n_context_dim,
                n_layers=n_layers,
                n_channels=n_channels,
                affine_activation=affine_activation,
                use_partial_padding=use_partial_padding,
            )
        elif affine_model == "simple_conv":
            self.affine_param_predictor = SimpleConvNet(
                int(n_mel_channels / 2),
                n_context_dim,
                n_mel_channels,
                n_layers,
                with_dilation=with_dilation,
                kernel_size=kernel_size,
                use_partial_padding=use_partial_padding,
            )
        self.n_mel_channels = n_mel_channels

    def get_scaling_and_logs(self, scale_unconstrained):
        if self.scaling_fn == "translate":
            s = torch.exp(scale_unconstrained * 0)
            log_s = scale_unconstrained * 0
        elif self.scaling_fn == "exp":
            s = torch.exp(scale_unconstrained)
            log_s = scale_unconstrained  # log(exp
        elif self.scaling_fn == "tanh":
            s = torch.tanh(scale_unconstrained) + 1 + 1e-6
            log_s = torch.log(s)
        elif self.scaling_fn == "sigmoid":
            s = torch.sigmoid(scale_unconstrained + 10) + 1e-6
            log_s = torch.log(s)
        elif isinstance(self.scaling_fn, list):
            s_list, log_s_list = [], []
            for i in range(scale_unconstrained.shape[1]):
                scaling_i = self.scaling_fn[i]
                if scaling_i == "translate":
                    s_i = torch.exp(scale_unconstrained[:i] * 0)
                    log_s_i = scale_unconstrained[:, i] * 0
                elif scaling_i == "exp":
                    s_i = torch.exp(scale_unconstrained[:, i])
                    log_s_i = scale_unconstrained[:, i]
                elif scaling_i == "tanh":
                    s_i = torch.tanh(scale_unconstrained[:, i]) + 1 + 1e-6
                    log_s_i = torch.log(s_i)
                elif scaling_i == "sigmoid":
                    s_i = torch.sigmoid(scale_unconstrained[:, i]) + 1e-6
                    log_s_i = torch.log(s_i)
                s_list.append(s_i[:, None])
                log_s_list.append(log_s_i[:, None])
            s = torch.cat(s_list, dim=1)
            log_s = torch.cat(log_s_list, dim=1)
        return s, log_s

    def forward(self, z, context, inverse=False, seq_lens=None):
        n_half = int(self.n_mel_channels / 2)
        z_0, z_1 = z[:, :n_half], z[:, n_half:]
        if self.affine_model == "wavenet":
            affine_params = self.affine_param_predictor(
                (z_0, context), seq_lens=seq_lens
            )
        elif self.affine_model == "simple_conv":
            z_w_context = torch.cat((z_0, context), 1)
            affine_params = self.affine_param_predictor(z_w_context, seq_lens=seq_lens)

        scale_unconstrained = affine_params[:, :n_half, :]
        b = affine_params[:, n_half:, :]
        s, log_s = self.get_scaling_and_logs(scale_unconstrained)

        if inverse:
            z_1 = (z_1 - b) / s
            z = torch.cat((z_0, z_1), dim=1)
            return z
        else:
            z_1 = s * z_1 + b
            z = torch.cat((z_0, z_1), dim=1)
            return z, log_s


class ConvAttention(torch.nn.Module):
    def __init__(
        self, n_mel_channels=80, n_text_channels=512, n_att_channels=80, temperature=1.0
    ):
        super(ConvAttention, self).__init__()
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=3)
        self.log_softmax = torch.nn.LogSoftmax(dim=3)

        self.key_proj = nn.Sequential(
            ConvNorm(
                n_text_channels,
                n_text_channels * 2,
                kernel_size=3,
                bias=True,
                w_init_gain="relu",
            ),
            torch.nn.ReLU(),
            ConvNorm(n_text_channels * 2, n_att_channels, kernel_size=1, bias=True),
        )

        self.query_proj = nn.Sequential(
            ConvNorm(
                n_mel_channels,
                n_mel_channels * 2,
                kernel_size=3,
                bias=True,
                w_init_gain="relu",
            ),
            torch.nn.ReLU(),
            ConvNorm(n_mel_channels * 2, n_mel_channels, kernel_size=1, bias=True),
            torch.nn.ReLU(),
            ConvNorm(n_mel_channels, n_att_channels, kernel_size=1, bias=True),
        )

    def run_padded_sequence(
        self, sorted_idx, unsort_idx, lens, padded_data, recurrent_model
    ):
        """Sorts input data by previded ordering (and un-ordering) and runs the
        packed data through the recurrent model

        Args:
            sorted_idx (torch.tensor): 1D sorting index
            unsort_idx (torch.tensor): 1D unsorting index (inverse of sorted_idx)
            lens: lengths of input data (sorted in descending order)
            padded_data (torch.tensor): input sequences (padded)
            recurrent_model (nn.Module): recurrent model to run data through
        Returns:
            hidden_vectors (torch.tensor): outputs of the RNN, in the original,
            unsorted, ordering
        """

        # sort the data by decreasing length using provided index
        # we assume batch index is in dim=1
        padded_data = padded_data[:, sorted_idx]
        padded_data = nn.utils.rnn.pack_padded_sequence(padded_data, lens)
        hidden_vectors = recurrent_model(padded_data)[0]
        hidden_vectors, _ = nn.utils.rnn.pad_packed_sequence(hidden_vectors)
        # unsort the results at dim=1 and return
        hidden_vectors = hidden_vectors[:, unsort_idx]
        return hidden_vectors

    def forward(
        self, queries, keys, query_lens, mask=None, key_lens=None, attn_prior=None
    ):
        """Attention mechanism for radtts. Unlike in Flowtron, we have no
        restrictions such as causality etc, since we only need this during
        training.

        Args:
            queries (torch.tensor): B x C x T1 tensor (likely mel data)
            keys (torch.tensor): B x C2 x T2 tensor (text data)
            query_lens: lengths for sorting the queries in descending order
            mask (torch.tensor): uint8 binary mask for variable length entries
                                 (should be in the T2 domain)
        Output:
            attn (torch.tensor): B x 1 x T1 x T2 attention mask.
                                 Final dim T2 should sum to 1
        """
        temp = 0.0005
        keys_enc = self.key_proj(keys)  # B x n_attn_dims x T2
        # Beware can only do this since query_dim = attn_dim = n_mel_channels
        queries_enc = self.query_proj(queries)

        # Gaussian Isotopic Attention
        # B x n_attn_dims x T1 x T2
        attn = (queries_enc[:, :, :, None] - keys_enc[:, :, None]) ** 2

        # compute log-likelihood from gaussian
        eps = 1e-8
        attn = -temp * attn.sum(1, keepdim=True)
        if attn_prior is not None:
            attn = self.log_softmax(attn) + torch.log(attn_prior[:, None] + eps)

        attn_logprob = attn.clone()

        if mask is not None:
            attn.data.masked_fill_(mask.permute(0, 2, 1).unsqueeze(2), -float("inf"))

        attn = self.softmax(attn)  # softmax along T2
        return attn, attn_logprob


def update_params(config, params):
    for param in params:
        print(param)
        k, v = param.split("=")
        try:
            v = ast.literal_eval(v)
        except Exception as e:
            print(e)

        k_split = k.split(".")
        if len(k_split) > 1:
            parent_k = k_split[0]
            cur_param = [".".join(k_split[1:]) + "=" + str(v)]
            update_params(config[parent_k], cur_param)
        elif k in config and len(k_split) == 1:
            print(f"overriding {k} with {v}")
            config[k] = v
        else:
            print("{}, {} params not updated".format(k, v))
