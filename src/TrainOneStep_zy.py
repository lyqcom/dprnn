import sys

sys.path.append('../')

from src.data_test1 import DatasetGenerator
import mindspore.dataset as ds
from mindspore import Model, nn, context, save_checkpoint
# from mindspore.train.callback import LossMonitor, TimeMonitor, ModelCheckpoint, CheckpointConfig
# from network_define import WithLossCell
# from Loss_final1 import loss
from model_rnn import Dual_RNN_model
from generatorLoss import Generatorloss
from trainonestep import TrainOneStep
import time
import argparse
import os

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

def train(trainoneStep , data ,args):
    trainoneStep.set_train()
    trainoneStep.set_grad()
    tr_loader = data['tr_loader']
    cv_loader = data['cv_loader']

    step = tr_loader.get_dataset_size()

    for epoch in range(10):
        total_loss = 0
        j=0
        for data in tr_loader:
            mixture, len, source = [x for x in data]
            t0 = time.time()
            loss = trainoneStep(mixture, len, source)
            t1 = time.time()
            print("epoch[{}]({}/{}),loss:{:.4f},stepTime:{}".format(epoch + 1, j+1, step, loss.asnumpy(), t1 - t0))

            if j == (step//2):
                save_ckpt = os.path.join(args.save_folder , 'half{}_{}_DPRNN.ckpt'.format(epoch + 1, j))
                save_checkpoint(trainoneStep.network , save_ckpt)
            j=j+1
            total_loss += loss
        train_loss = total_loss/j
        print("epoch[{}]:trainAvgLoss:{:.4f}".format(epoch + 1, train_loss.asnumpy()))
        save_ckpt = os.path.join(args.save_folder, '{}_DPRNN.ckpt'.format(epoch + 1))
        save_checkpoint(trainoneStep.network, save_ckpt)

def main():
    args = parser.parse_args()
    # build dataloader
    tr_dataset = DatasetGenerator(args.train_dir, args.batch_size,
                                  sample_rate=args.sample_rate, segment=args.segment)
    tr_loader = ds.GeneratorDataset(tr_dataset, ["mixture", "lens", "sources"], shuffle=False)

    tr_loader = tr_loader.batch(batch_size=2)
    cv_dataset = DatasetGenerator(args.valid_dir, args.batch_size,
                                  sample_rate=args.sample_rate, segment=args.segment)
    cv_loader = ds.GeneratorDataset(cv_dataset, ["mixture", "lens", "sources"], shuffle=False)

    cv_loader = cv_loader.batch(batch_size=2)
    data = {"tr_loader": tr_loader, "cv_loader": cv_loader}

    # build model
    net = Dual_RNN_model(args.in_channels, args.out_channels, args.hidden_channels,
                         bidirectional=True, norm=args.norm, num_layers=args.num_layers, dropout=args.dropout, K=args.K)
    # print(net)
    # net.set_train()

    # loss
    loss_network = Generatorloss(net)

    optimizier = nn.Adam(net.trainable_params(), learning_rate=args.lr, beta1=0.9, beta2=0.999, weight_decay=args.l2)

    # 前向到loss
    trainonestepNet = TrainOneStep(loss_network, optimizier, sens=1.0)

    # trainonestepNet.set_train()

    train(trainonestepNet, data, args)

if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=3)
    main()
