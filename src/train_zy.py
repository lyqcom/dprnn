import sys

sys.path.append('../')

from data_test1 import DatasetGenerator
import mindspore.dataset as ds
from mindspore import Model, load_checkpoint, load_param_into_net
from mindspore import nn, context
from mindspore.train.callback import LossMonitor, TimeMonitor, ModelCheckpoint, CheckpointConfig
from network_define import WithLossCell
from Loss_final1 import loss
from model_rnn import Dual_RNN_model
import argparse
# from mindspore.profiler import Profiler

parser = argparse.ArgumentParser(
        description='Parameters for training Dual-Path-RNN')

parser.add_argument('--opt', type=str, help='Path to option YAML file.')
# parser.add_argument('--train_dir', type=str, default='/mass_data/dataset/LS-2mix/Libri2Mix/tr',
#                     help='directory including mix.json, s1.json and s2.json')
# parser.add_argument('--train_dir', type=str, default='/home/heu_MEDAI/zhangyu/project/out_dir/tr',
#                     help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--train_dir', type=str, default='/home/ma-user/work/DPRNN-mindspore/out_dir/tr',
                     help='directory including mix.json, s1.json and s2.json')
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
parser.add_argument('--num_layers', default=4, type=int,
                    help='Number of Dual-Path-Block')
parser.add_argument('--K', default=250, type=int,
                    help='The length of chunk')
parser.add_argument('--num_spks', default=2, type=int,
                    help='The number of speakers')

# optimizer
parser.add_argument('--optimizer', default='adam', type=str,
                    choices=['sgd', 'adam'],
                    help='Optimizer (support sgd and adam now)')
parser.add_argument('--lr', default=5e-4, type=float,
                    help='Init learning rate')
parser.add_argument('--momentum', default=0.0, type=float,
                    help='Momentum for optimizer')
parser.add_argument('--l2', default=0.01, type=float,
                    help='weight decay (L2 penalty)')

# save and load model
parser.add_argument('--save_folder', default='/home/ma-user/work/DPRNN-mindspore/checkpoint',
                    help='Location to save epoch models')
parser.add_argument('--checkpoint', dest='checkpoint', default=0, type=int,
                    help='Enables checkpoint saving of model')
parser.add_argument('--continue_from', default='',
                    help='Continue from checkpoint model')
parser.add_argument('--model_path', default='final.pth.tar',
                    help='Location to save best validation model')

def train():
    # profiler = Profiler(output_path='../profiler_data')
    args = parser.parse_args()
    # build dataloader
    tr_dataset = DatasetGenerator(args.train_dir, args.batch_size,
                                  sample_rate=args.sample_rate, segment=args.segment)
    tr_loader = ds.GeneratorDataset(tr_dataset, ["mixture", "lens", "sources"], shuffle=False)

    tr_loader = tr_loader.batch(batch_size=2)
    num_steps = tr_loader.get_dataset_size()
    # param_dict = load_checkpoint("/home/heu_MEDAI/zhangyu/project/checkpoint/DPRNN_ckpt_1-11_7120.ckpt")
    # build model
    net = Dual_RNN_model(args.in_channels, args.out_channels, args.hidden_channels,
                         bidirectional=True, norm=args.norm, num_layers=args.num_layers, dropout=args.dropout, K=args.K)
    print(net)
    # load_param_into_net(net, param_dict)
    net.set_train()
    # build optimizer
    optimizier = nn.Adam(net.get_parameters(), learning_rate=args.lr, beta1=0.9, beta2=0.98, eps=1e-9, weight_decay=args.l2)
    my_loss = loss()
    net_with_loss = WithLossCell(net, my_loss)
    model = Model(net_with_loss, optimizer=optimizier)

    # loss_cb = LossMonitor(10)
    loss_cb = LossMonitor(1)
    time_cb = TimeMonitor(data_size=num_steps)
    cb = [time_cb, loss_cb]
    
    config_ck = CheckpointConfig(save_checkpoint_steps=20, keep_checkpoint_max=2)
    ckpt_cb = ModelCheckpoint(prefix='DPRNN',
                              directory=args.save_folder,
                              config=config_ck)
    cb += [ckpt_cb]

    model.train(epoch=10, train_dataset=tr_loader, callbacks=cb, dataset_sink_mode=False)
    # profiler.analyse()

if __name__ == "__main__":
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend", device_id=0)
    train()
