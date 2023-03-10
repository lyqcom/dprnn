
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
'''
##############evaluate trained models#################
python export.py
'''

import numpy as np
from mindspore.train.serialization import export
from mindspore import Tensor, context
from mindspore.train.serialization import load_checkpoint, load_param_into_net

#from model_final import DPTNet_base
from model_rnn import Dual_RNN_model
import argparse

parser = argparse.ArgumentParser(
        description='Parameters for training Dual-Path-RNN')

parser.add_argument('--opt', type=str, help='Path to option YAML file.')
parser.add_argument('--train_dir', type=str, default='/mass_data/dataset/LS-2mix/Libri2Mix/tr',
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--valid_dir', type=str, default='/mass_data/dataset/LS-2mix/Libri2Mix/cv',
                    help='directory including mix.json, s1.json and s2.json')
# parser.add_argument('--train_dir', type=str, default='/home/heu_MEDAI/zhangyu/project/out_dir/tr',
#                     help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--sample_rate', default=8000, type=int,
                    help='Sample rate')
parser.add_argument('--segment', default=4, type=float,  # 取音频的长度，2s。#数据集语音长度要相同
                    help='Segment length (seconds)')
parser.add_argument('--batch_size', default=2, type=int,  # 需要抛弃的音频长度
                    help='Batch size')

# Network architecture
parser.add_argument('--in_channels', default=256, type=int,
                    help='The number of expected features in the input')
parser.add_argument('--out_channels', default=64, type=int,
                    help='The number of features in the hidden state')
parser.add_argument('--hidden_channels', default=128, type=int,
                    help='The hidden size of RNN')
parser.add_argument('--kernel_size', default=2, type=int,
                    help='Encoder and Decoder Kernel size')
parser.add_argument('--rnn_type', default='LSTM', type=str,
                    help='RNN, LSTM, GRU')
parser.add_argument('--norm', default='ln', type=str,
                    help='gln = "Global Norm", cln = "Cumulative Norm", ln = "Layer Norm"')
parser.add_argument('--dropout', default=0.0, type=float,
                    help='dropout')
parser.add_argument('--num_layers', default=6, type=int,
                    help='Number of Dual-Path-Block')
parser.add_argument('--K', default=250, type=int,
                    help='The length of chunk')
parser.add_argument('--num_spks', default=2, type=int,
                    help='The number of speakers')

# optimizer
# parser.add_argument('--optimizer', default='adam', type=str,
#                     choices=['sgd', 'adam'],
#                     help='Optimizer (support sgd and adam now)')
parser.add_argument('--lr', default=5e-4, type=float,
                    help='Init learning rate')
parser.add_argument('--momentum', default=0.0, type=float,
                    help='Momentum for optimizer')
parser.add_argument('--l2', default=0.01, type=float,
                    help='weight decay (L2 penalty)')

# save and load model
parser.add_argument('--save_folder', default='/home/heu_MEDAI/zhangyu/project/onestepCkpt',
                    help='Location to save epoch models')
parser.add_argument('--checkpoint', dest='checkpoint', default=0, type=int,
                    help='Enables checkpoint saving of model')
parser.add_argument('--continue_from', default='',
                    help='Continue from checkpoint model')
parser.add_argument('--model_path', default='final.pth.tar',
                    help='Location to save best validation model')

def export_DPTNet():
    """ export_fcn4 """
    args = parser.parse_args()
    net = Dual_RNN_model(args.in_channels, args.out_channels, args.hidden_channels,
                         bidirectional=True, norm=args.norm, num_layers=args.num_layers, dropout=args.dropout, K=args.K)

    param_dict = load_checkpoint("/home/heu_MEDAI/zhangyu/project/onestepCkpt/8_DPRNN.ckpt")
    load_param_into_net(net, param_dict)
    input_data = Tensor(np.random.uniform(0.0, 1.0, size=[2, 32000]).astype(np.float32))
    export(net, input_data, file_name='dprnn', file_format='MINDIR')
    print("export success")

if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=4)
    export_DPTNet()
