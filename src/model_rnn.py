from re import T
import sys

sys.path.append('/')

import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, context, Tensor
import mindspore.common.initializer
from util import check_parameters
import warnings

warnings.filterwarnings('ignore')

class GlobalLayerNorm(nn.Cell):
    '''
       Calculate Global Layer Normalization
       dim: (int or list or torch.Size) –
          input shape from an expected input of size
       eps: a value added to the denominator for numerical stability.
       elementwise_affine: a boolean value that when set to True, 
          this module has learnable per-element affine parameters 
          initialized to ones (for weights) and zeros (for biases).
    '''

    def __init__(self, dim, shape, eps=1e-8, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.ones = ops.Ones()
        self.zeros = ops.Zeros()
        self.mean = ops.ReduceMean(keep_dims=True)
        self.sqrt = ops.Sqrt()

        if self.elementwise_affine:
            if shape == 3:
                self.weight = Parameter(self.ones((self.dim, 1), mindspore.float32), name="w1")
                self.bias = Parameter(self.zeros((self.dim, 1), mindspore.float32), name="w2")
            if shape == 4:
                self.weight = Parameter(self.ones((self.dim, 1, 1), mindspore.float32), name="w3")
                self.bias = Parameter(self.zeros((self.dim, 1, 1), mindspore.float32), name="w4")
        else:
            self.insert_param_to_cell('weight', None)
            self.insert_param_to_cell('bias', None)

    def construct(self, x):
        # x = N x C x K x S or N x C x L
        # N x 1 x 1
        # cln: mean,var N x 1 x K x S
        # gln: mean,var N x 1 x 1
        if x.ndim == 4:
            mean = self.mean(x, (1, 2, 3))
            var = self.mean((x-mean)**2, (1, 2, 3))
            if self.elementwise_affine:
                x = self.weight*(x-mean)/self.sqrt(var+self.eps)+self.bias
            else:
                x = (x-mean)/self.sqrt(var+self.eps)
        if x.ndim == 3:
            mean = self.mean(x, (1, 2))
            var = self.mean((x-mean)**2, (1, 2))
            if self.elementwise_affine:
                x = self.weight*(x-mean)/self.sqrt(var+self.eps)+self.bias
            else:
                x = (x-mean)/self.sqrt(var+self.eps)
        return x


class CumulativeLayerNorm(nn.LayerNorm):
    '''
       Calculate Cumulative Layer Normalization
       dim: you want to norm dim
    '''

    def __init__(self, dim):
        super(CumulativeLayerNorm, self).__init__(dim, epsilon=1e-8)
        self.transpose = ops.Transpose()

    def construct(self, x):
        # x: N x C x K x S or N x C x L
        # N x K x S x C
        if x.ndim == 4:
           x = x.transpose((0, 2, 3, 1))
           # N x K x S x C == only channel norm
           x = super().construct(x)
           # N x C x K x S
           x = x.transpose((0, 3, 1, 2))
        if x.ndim == 3:
            x = self.transpose(x, (0, 2, 1))
            # N x L x C == only channel norm
            x = super().construct(x)
            # N x C x L
            x = self.transpose(x, (0, 2, 1))
        return x


def select_norm(norm, dim, shape):
    if norm == 'gln':
        return GlobalLayerNorm(dim, shape, elementwise_affine=True)
    if norm == 'cln':
        return CumulativeLayerNorm(dim)
    if norm == 'ln':
        return nn.GroupNorm(1, dim, eps=1e-8, affine=True)
    else:
        return nn.BatchNorm1d(dim)

class Encoder(nn.Cell):
    '''
       Conv-Tasnet Encoder part
       kernel_size: the length of filters
       out_channels: the number of filters
    '''

    def __init__(self, kernel_size=2, out_channels=64):
        super(Encoder, self).__init__()
        self.expand_dims = ops.ExpandDims()
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=out_channels, kernel_size=kernel_size,
                                stride=kernel_size//2, weight_init="HeUniform", pad_mode="pad")
        self.relu = ops.ReLU()

    def construct(self, x):
        """
          Input:
              x: [B, T], B is batch size, T is times
          Returns:
              x: [B, C, T_out]
              T_out is the number of time steps
        """
        # B x T -> B x 1 x T
        x = self.expand_dims(x, 1)
        # B x 1 x T -> B x C x T_out
        x = self.conv1d(x)
        x = self.relu(x)
        return x


class Decoder(nn.Cell):
    '''
        Decoder of the TasNet
        This module can be seen as the gradient of Conv1d with respect to its input. 
        It is also known as a fractionally-strided convolution 
        or a deconvolution (although it is not an actual deconvolution operation).
    '''

    def __init__(self, in_channels=256, kernel_size=2):
        super(Decoder, self).__init__()
        self.conv1dTranspose = nn.Conv1dTranspose(in_channels=in_channels, out_channels=1, kernel_size=kernel_size,
                                                  stride=kernel_size//2, weight_init='HeUniform', pad_mode='pad')
        self.expand_dims = ops.ExpandDims()
        self.squeeze = ops.Squeeze()

    def construct(self, x):
        """
        x: [B, N, L]
        """

        # if x.ndim not in [2, 3]:
        #     raise RuntimeError("{} accept 3/4D tensor as input".format(
        #         self.__name__))
        x = self.conv1dTranspose(x if x.ndim == 3 else self.expand_dims(x, 1))

        if self.squeeze(x).ndim == 1:
            squeeze1 = ops.Squeeze(axis=1)
            x = squeeze1(x)
        else:
            x = self.squeeze(x)
        return x


class Dual_RNN_Block(nn.Cell):
    '''
       Implementation of the intra-RNN and the inter-RNN
       input:
            in_channels: The number of expected features in the input x
            out_channels: The number of features in the hidden state h
            rnn_type: RNN, LSTM, GRU
            norm: gln = "Global Norm", cln = "Cumulative Norm", ln = "Layer Norm"
            dropout: If non-zero, introduces a Dropout layer on the outputs 
                     of each LSTM layer except the last layer, 
                     with dropout probability equal to dropout. Default: 0
            bidirectional: If True, becomes a bidirectional LSTM. Default: False
    '''

    def __init__(self, out_channels,
                 hidden_channels, rnn_type='LSTM', norm='ln',
                 dropout=0, bidirectional=False, num_spks=2):
        super(Dual_RNN_Block, self).__init__()
        # RNN model
        self.intra_rnn = getattr(nn, rnn_type)(
            out_channels, hidden_channels, 1, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.h0 = Tensor(np.ones([2 * 1, 516, hidden_channels]).astype(np.float32))
        self.c0 = Tensor(np.ones([2 * 1, 516, hidden_channels]).astype(np.float32))
        self.inter_rnn = getattr(nn, rnn_type)(
            out_channels, hidden_channels, 1, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.h1 = Tensor(np.ones([2 * 1, 500, hidden_channels]).astype(np.float32))
        self.c1 = Tensor(np.ones([2 * 1, 500, hidden_channels]).astype(np.float32))
        
        # Norm
        self.intra_norm = select_norm(norm, out_channels, 4)
        self.inter_norm = select_norm(norm, out_channels, 4)
        # Linear
        self.intra_linear = nn.Dense(
            hidden_channels*2 if bidirectional else hidden_channels, out_channels, weight_init="XavierUniform")
        self.inter_linear = nn.Dense(
            hidden_channels*2 if bidirectional else hidden_channels, out_channels, weight_init="XavierUniform")
        

    def construct(self, x):
        '''
           x: [B, N, K, S]
           out: [Spks, B, N, K, S]
        '''
        B, N, K, S = x.shape
        # intra RNN
        # [BS, K, N]
        intra_rnn = x.transpose((0, 3, 2, 1)).view(B*S, K, N)
        # [BS, K, H]
        # intra_rnn, _ = self.intra_rnn(intra_rnn)
        intra_rnn, (hn, cn) = self.intra_rnn(intra_rnn, (self.h0, self.c0))
        # [BS, K, N]
        intra_rnn = self.intra_linear(intra_rnn.view(B*S*K, -1)).view(B*S, K, -1)
        # [B, S, K, N]
        intra_rnn = intra_rnn.view(B, S, K, N)
        # [B, N, K, S]
        intra_rnn = intra_rnn.transpose((0, 3, 2, 1))
        intra_rnn = self.intra_norm(intra_rnn)
        
        # [B, N, K, S]
        intra_rnn = intra_rnn + x

        # inter RNN
        # [BK, S, N]
        inter_rnn = intra_rnn.transpose((0, 2, 3, 1)).view(B*K, S, N)
        # [BK, S, H]
        # inter_rnn, _ = self.inter_rnn(inter_rnn)
        inter_rnn, (hn, cn) = self.inter_rnn(inter_rnn, (self.h1, self.c1))
        # [BK, S, N]
        inter_rnn = self.inter_linear(inter_rnn.view(B*S*K, -1)).view(B*K, S, -1)
        # [B, K, S, N]
        inter_rnn = inter_rnn.view(B, K, S, N)
        # [B, N, K, S]
        inter_rnn = inter_rnn.transpose((0, 3, 1, 2))
        inter_rnn = self.inter_norm(inter_rnn)
        # [B, N, K, S]
        out = inter_rnn + intra_rnn

        return out


class Dual_Path_RNN(nn.Cell):
    '''
       Implementation of the Dual-Path-RNN model
       input:
            in_channels: The number of expected features in the input x
            out_channels: The number of features in the hidden state h
            rnn_type: RNN, LSTM, GRU
            norm: gln = "Global Norm", cln = "Cumulative Norm", ln = "Layer Norm"
            dropout: If non-zero, introduces a Dropout layer on the outputs 
                     of each LSTM layer except the last layer, 
                     with dropout probability equal to dropout. Default: 0
            bidirectional: If True, becomes a bidirectional LSTM. Default: False
            num_layers: number of Dual-Path-Block
            K: the length of chunk
            num_spks: the number of speakers
    '''

    def __init__(self, in_channels, out_channels, hidden_channels,
                 rnn_type='LSTM', norm='ln', dropout=0,
                 bidirectional=False, num_layers=4, K=200, num_spks=2):
        super(Dual_Path_RNN, self).__init__()
        self.K = K
        self.selectNorm = norm
        self.num_spks = num_spks
        self.num_layers = num_layers
        self.norm = select_norm(norm, in_channels, 3)
        self.norm1 = select_norm(norm, in_channels, 4)
        self.expand_dims = ops.ExpandDims()
        self.conv1d = nn.Conv1d(in_channels, out_channels, 1, weight_init='HeUniform')

        self.dual_rnn = nn.CellList([])
        for i in range(num_layers):
            self.dual_rnn.append(Dual_RNN_Block(out_channels, hidden_channels,
                                     rnn_type=rnn_type, norm=norm, dropout=dropout,
                                     bidirectional=bidirectional))
        self.zeros = ops.Zeros()
        self.concat_op2 = ops.Concat(2)
        self.concat_op3 = ops.Concat(3)
        self.conv2d = nn.Conv2d(
            out_channels, out_channels*num_spks, kernel_size=1, weight_init='HeUniform')
        self.end_conv1x1 = nn.Conv1d(out_channels, in_channels, 1, weight_init='HeUniform')
        self.prelu = nn.PReLU()
        self.activation = nn.ReLU()
         # gated output layer
        self.output = nn.SequentialCell([nn.Conv1d(out_channels, out_channels, 1),
                                         nn.Tanh()])
        self.output_gate = nn.SequentialCell([nn.Conv1d(out_channels, out_channels, 1),
                                              nn.Sigmoid()])

    def construct(self, x):
        '''
           x: [B, N, L]

        '''
        # [B, N, L]
        if(self.selectNorm == 'ln'):
            x = self.expand_dims(x, 0).transpose((0, 2, 1, 3))
            x = self.norm1(x)
            x = x.transpose((0, 2, 1, 3)).squeeze(axis=0)
        else:
            x = self.norm(x)
        # [B, N, L]
        x = self.conv1d(x)
        # [B, N, K, S]
        x, gap = self._Segmentation(x, self.K)
        # [B, N*spks, K, S]
        for i in range(self.num_layers):
            x = self.dual_rnn[i](x)
        x = self.prelu(x)
        x = self.conv2d(x)
        # [B*spks, N, K, S]
        B, _, K, S = x.shape
        x = x.view(B*self.num_spks, -1, K, S)
        # [B*spks, N, L]
        x = self._over_add(x, gap)
        x = self.output(x) * self.output_gate(x)
        # [spks*B, N, L]
        x = self.end_conv1x1(x)
        # [B*spks, N, L] -> [B, spks, N, L]
        _, N, L = x.shape
        x = x.view(B, self.num_spks, N, L)
        x = self.activation(x)
        # [spks, B, N, L]
        x = x.transpose(1, 0, 2, 3)

        return x

    def _padding(self, input, K):
        '''
           padding the audio times
           K: chunks of length
           P: hop size
           input: [B, N, L]
        '''
        B, N, L = input.shape
        P = K // 2
        gap = K - (P + L % K) % K

        if gap > 0:
            pad = self.zeros((B, N, gap), mindspore.float32)
            input = self.concat_op2([input, pad])

        # _pad = Tensor(torch.zeros(B, N, P)).type(input.type())
        _pad = self.zeros((B, N, P), mindspore.float32)
        input = self.concat_op2([_pad, input, _pad])

        return input, gap

    def _Segmentation(self, input, K):
        '''
           the segmentation stage splits
           K: chunks of length
           P: hop size
           input: [B, N, L]
           output: [B, N, K, S]
        '''
        B, N, L = input.shape
        P = K // 2
        input, gap = self._padding(input, K)
        # [B, N, K, S]
        input1 = input[:, :, :-P].view(B, N, -1, K)
        input2 = input[:, :, P:].view(B, N, -1, K)

        input = self.concat_op3([input1, input2]).view(
            B, N, -1, K).transpose((0, 1, 3, 2))

        return input, gap

    def _over_add(self, input, gap):
        '''
           Merge sequence
           input: [B, N, K, S]
           gap: padding length
           output: [B, N, L]
        '''
        B, N, K, S = input.shape
        P = K // 2
        # [B, N, S, K]
        input = input.transpose((0, 1, 3, 2)).view(B, N, -1, K * 2)

        input1 = input[:, :, :, :K].view(B, N, -1)[:, :, P:]
        input2 = input[:, :, :, K:].view(B, N, -1)[:, :, :-P]
        input = input1 + input2
        # [B, N, L]
        if gap > 0:
            input = input[:, :, :-gap]

        return input

class Dual_RNN_model(nn.Cell):
    '''
       model of Dual Path RNN
       input:
            in_channels: The number of expected features in the input x
            out_channels: The number of features in the hidden state h
            hidden_channels: The hidden size of RNN
            kernel_size: Encoder and Decoder Kernel size
            rnn_type: RNN, LSTM, GRU
            norm: gln = "Global Norm", cln = "Cumulative Norm", ln = "Layer Norm"
            dropout: If non-zero, introduces a Dropout layer on the outputs 
                     of each LSTM layer except the last layer, 
                     with dropout probability equal to dropout. Default: 0
            bidirectional: If True, becomes a bidirectional LSTM. Default: False
            num_layers: number of Dual-Path-Block
            K: the length of chunk
            num_spks: the number of speakers
    '''
    def __init__(self, in_channels, out_channels, hidden_channels,
                 kernel_size=2, rnn_type='LSTM', norm='ln', dropout=0,
                 bidirectional=False, num_layers=4, K=200, num_spks=2):
        super(Dual_RNN_model, self).__init__()
        self.encoder = Encoder(kernel_size=kernel_size, out_channels=in_channels)
        self.separation = Dual_Path_RNN(in_channels, out_channels, hidden_channels,
                 rnn_type=rnn_type, norm=norm, dropout=dropout,
                 bidirectional=bidirectional, num_layers=num_layers, K=K, num_spks=num_spks)
        self.decoder = Decoder(in_channels=in_channels, kernel_size=kernel_size)
        self.num_spks = num_spks
        self.print = ops.Print()
        self.stack = ops.Stack()

        for p in self.get_parameters():
            if p.ndim > 1:
                mindspore.common.initializer.XavierUniform(p)
        #         mindspore.common.initializer.HeNormal(p)
    def construct(self, x):
        """ forward """
        '''
           x: [B, L]
        '''
        # [B, N, L]
        e = self.encoder(x)
        # [spks, B, N, L]
        s = self.separation(e)
        # [B, N, L] -> [B, L]
        # out = [s[i]*e for i in range(self.num_spks)]
        audio = []
        for i in range(self.num_spks):
            audio.append(self.decoder(s[i] * e))
        audio = self.stack(audio)
        audio = audio.transpose((1, 0, 2))
        return audio

if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=0)
    rnn = Dual_RNN_model(256, 64, 128, bidirectional=True, norm='ln', num_layers=6, dropout=0.0)
    #encoder = Encoder(16, 512)
    ones = ops.Ones()
    x = ones((1, 100), mindspore.float32)
    out = rnn(x)
    print("{:.3f}".format(check_parameters(rnn)*1000000))
    print(rnn)
