import sys

sys.path.append('../')

from src.data_test import DatasetGenerator
import mindspore.dataset as ds
from mindspore import Model
from mindspore import nn
from mindspore.nn import Accuracy
from mindspore.train.callback import Callback, LossMonitor, TimeMonitor, ModelCheckpoint, CheckpointConfig
from network_define import WithLossCell
from Loss_final1 import loss
from model_rnn import Dual_RNN_model
from logger import set_logger
import logging
from config import option
import argparse

parser = argparse.ArgumentParser(
        description='Parameters for training Dual-Path-RNN')
# set_logger
parser.add_argument('--log_name', default='DPCL', type=str,
                    help='Sample rate')
parser.add_argument('--log_path', default='/home/heu_MEDAI/zhangyu/project/logger', type=str,
                    help='Sample rate')
parser.add_argument('--screen', default=1, type=int,
                    help='Sample rate')
parser.add_argument('--tofile', default=0, type=int,
                    help='Sample rate')

parser.add_argument('--opt', type=str, help='Path to option YAML file.')
# parser.add_argument('--train_dir', type=str, default='/mass_data/dataset/LS-2mix/Libri2Mix/tr',
#                     help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--train_dir', type=str, default='/home/heu_MEDAI/zhangyu/project/out_dir/tr',
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--val_dir', type=str, default='/home/heu_MEDAI/zhangyu/project/out_dir/cv',
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
parser.add_argument('--num_layers', default=6, type=int,
                    help='Number of Dual-Path-Block')
parser.add_argument('--K', default=250, type=int,
                    help='The length of chunk')
parser.add_argument('--num_spks', default=2, type=int,
                    help='The number of speakers')

# optimizer
parser.add_argument('--lr', default=5e-4, type=float,
                    help='Init learning rate')
parser.add_argument('--l2', default=0.0, type=float,
                    help='weight decay (L2 penalty)')

# save and load model
parser.add_argument('--save_folder', default='/home/heu_MEDAI/zhangyu/project/checkpoint',
                    help='Location to save epoch models')
parser.add_argument('--checkpoint', dest='checkpoint', default=0, type=int,
                    help='Enables checkpoint saving of model')
parser.add_argument('--continue_from', default='',
                    help='Continue from checkpoint model')
parser.add_argument('--model_path', default='final.pth.tar',
                    help='Location to save best validation model')

class EvalCallBack(Callback):
    """Precision verification using callback function."""
    # define the operator required
    def __init__(self, models, eval_dataset, eval_per_epochs, epochs_per_eval):
        super(EvalCallBack, self).__init__()
        self.models = models
        self.eval_dataset = eval_dataset
        self.eval_per_epochs = eval_per_epochs
        self.epochs_per_eval = epochs_per_eval

    # define operator function in epoch end
    def epoch_end(self, run_context):
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        if cur_epoch % self.eval_per_epochs == 0:
            acc = self.models.eval(self.eval_dataset, dataset_sink_mode=False)
            self.epochs_per_eval["epoch"].append(cur_epoch)
            self.epochs_per_eval["acc"].append(acc["Accuracy"])
            print(acc)

def train():

    args = parser.parse_args()
    set_logger.setup_logger(args.log_name, args.log_path, screen=bool(args.screen), tofile=bool(args.tofile))
    # build optimizer
    logger = logging.getLogger(args.log_name)
    # build dataloader
    logger.info('Building the dataloader of Dual-Path-RNN')
    tr_dataset = DatasetGenerator(args.train_dir, args.batch_size,
                                  sample_rate=args.sample_rate, segment=args.segment)
    cv_dataset = DatasetGenerator(args.train_dir, args.batch_size,
                                  sample_rate=args.sample_rate, segment=args.segment)
    # tr_loader = ds.GeneratorDataset(tr_dataset, ["mixture", "lens", "sources"], shuffle=False)
    # cv_loader = ds.GeneratorDataset(cv_dataset, ["mixture", "lens", "sources"], shuffle=False)
    tr_loader = ds.GeneratorDataset(tr_dataset, ["mixture", "lens", "sources"], shuffle=False)
    cv_loader = ds.GeneratorDataset(cv_dataset, ["mixture", "lens", "sources"], shuffle=False)
    tr_loader = tr_loader.batch(batch_size=2)
    cv_loader = cv_loader.batch(batch_size=2)
    # build model
    logger.info("Building the model of Dual-Path-RNN")
    net = Dual_RNN_model(args.in_channels, args.out_channels, args.hidden_channels,
                         bidirectional=True, norm=args.norm, num_layers=args.num_layers, dropout=args.dropout, K=args.K)
    print(net)
    net = net.set_train()
    logger.info("Building the optimizer of Dual-Path-RNN")
    optimizier = nn.Adam(net.get_parameters(), learning_rate=args.lr, weight_decay=args.l2)
    my_loss = loss()
    # loss_cb = LossMonitor(500)
    loss_cb = LossMonitor()
    num_steps = tr_loader.get_dataset_size()
    time_cb = TimeMonitor(data_size=num_steps)
    net_with_loss = WithLossCell(net, my_loss)
    config_ck = CheckpointConfig(save_checkpoint_steps=2*num_steps, keep_checkpoint_max=1)
    ckpt_cb = ModelCheckpoint(prefix='DPRNN_ckpt',
                              directory=args.save_folder,
                              config=config_ck)

    logger.info('Building the Trainer of Dual-Path-RNN')
    model = Model(net, net_with_loss, optimizer=optimizier, metrics={"Accuracy": Accuracy()})
    epoch_per_eval = {"epoch": [], "acc": []}
    eval_cb = EvalCallBack(model, cv_loader, 2, epoch_per_eval)
    cb = [time_cb, loss_cb, ckpt_cb, eval_cb]

    model.train(epoch=10, train_dataset=tr_loader, callbacks=cb, dataset_sink_mode=False)
    # logger.info('Train Datasets Length: {}, Val Datasets Length: {}'.format(
    #     len(train_dataloader), len(val_dataloader)))
    # # build scheduler
    # scheduler = ReduceLROnPlateau(
    #     optimizer, mode='min',
    #     factor=opt['scheduler']['factor'],
    #     patience=opt['scheduler']['patience'],
    #     verbose=True, min_lr=opt['scheduler']['min_lr'])
    #
    # # build trainer
    # logger.info('Building the Trainer of Dual-Path-RNN')
    # trainer = trainer_Dual_RNN.Trainer(train_dataloader, val_dataloader, Dual_Path_RNN, optimizer, scheduler, opt)
    # trainer.run()


if __name__ == "__main__":
    from mindspore import context
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=7)
    train()
