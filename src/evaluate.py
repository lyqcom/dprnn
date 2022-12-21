
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

import argparse
import numpy as np
from model_rnn import Dual_RNN_model
from mir_eval.separation import bss_eval_sources
from data_test1 import DatasetGenerator
from Loss_final1 import loss
from function import get_input_with_list
import mindspore
import mindspore.dataset as ds
import mindspore.ops as ops
from mindspore import context
from mindspore import load_checkpoint, load_param_into_net

parser = argparse.ArgumentParser('Evaluate separation performance using TasNet')
parser.add_argument('--model_path', type=str,
                    default=r"/home/heu_MEDAI/zhangyu/project/checkpoint/DPRNN_ckpt-10_2395.ckpt",
                    help='Path to model file created by training')
parser.add_argument('--data_dir', type=str,
                    default=r"/mass_data/dataset/LS-2mix/Libri2Mix/tt",
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--cal_sdr', type=int, default=1,
                    help='Whether calculate SDR, add this option because calculation of SDR is very slow')
parser.add_argument('--use_cuda', type=int, default=0,
                    help='Whether use GPU')
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
parser.add_argument('--num_layers', default=1, type=int,
                    help='Number of Dual-Path-Block')
parser.add_argument('--K', default=250, type=int,
                    help='The length of chunk')
parser.add_argument('--num_spks', default=2, type=int,
                    help='The number of speakers')

def evaluate(args):
    total_SISNRi = 0
    total_SDRi = 0
    total_cnt = 0

    # Load model
    model = Dual_RNN_model(args.in_channels, args.out_channels, args.hidden_channels,
                         bidirectional=True, norm=args.norm, num_layers=args.num_layers, dropout=args.dropout, K=args.K)
    model.set_train(mode=False)
    param_dict = load_checkpoint(args.model_path)
    load_param_into_net(model, param_dict)
    print(model)

    # Load data
    tt_dataset = DatasetGenerator(args.data_dir, args.batch_size,
                                  sample_rate=args.sample_rate, segment=args.segment)
    tt_loader = ds.GeneratorDataset(tt_dataset, ["mixture", "lens", "sources"], shuffle=False)
    tt_loader = tt_loader.batch(batch_size=1)

    for data in tt_loader.create_dict_iterator():
        padded_mixture = data["mixture"]
        mixture_lengths = data["lens"]
        padded_source = data["sources"]
        padded_mixture = ops.Cast()(padded_mixture, mindspore.float32)
        padded_source = ops.Cast()(padded_source, mindspore.float32)
        # mixture_lengths_with_list = get_input_with_list(args.data_dir)
        mixture_lengths_with_list = mixture_lengths.asnumpy().tolist()
        estimate_source = model(padded_mixture)  # [B, C, T]

        my_loss = loss()
        cal_loss, max_snr, estimate_source, reorder_estimate_source = \
            my_loss(padded_source, estimate_source, mixture_lengths)
        # Remove padding and flat
        mixture = remove_pad(padded_mixture, mixture_lengths_with_list)
        source = remove_pad(padded_source, mixture_lengths_with_list)
        # NOTE: use reorder estimate source
        estimate_source = remove_pad(reorder_estimate_source,
                                     mixture_lengths_with_list)
        # for each utterance
        for mix, src_ref, src_est in zip(mixture, source, estimate_source):
            print("Utt", total_cnt + 1)
            # Compute SDRi
            if args.cal_sdr:
                avg_SDRi = cal_SDRi(src_ref, src_est, mix)
                total_SDRi += avg_SDRi
                print("\tSDRi={0:.2f}".format(-avg_SDRi))
            # Compute SI-SNRi
            avg_SISNRi = cal_SISNRi(src_ref, src_est, mix)
            print("\tSI-SNRi={0:.2f}".format(-avg_SISNRi))
            total_SISNRi += avg_SISNRi
            total_cnt += 1
    if args.cal_sdr:
        print("Average SDR improvement: {0:.2f}".format(-total_SDRi / total_cnt))
    print("Average SISNR improvement: {0:.2f}".format(-total_SISNRi / total_cnt))


def cal_SDRi(src_ref, src_est, mix):
    """Calculate Source-to-Distortion Ratio improvement (SDRi).
    NOTE: bss_eval_sources is very very slow.
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T]
    Returns:
        average_SDRi
    """
    src_anchor = np.stack([mix, mix], axis=0)
    sdr, sir, sar, popt = bss_eval_sources(src_ref, src_est)
    sdr0, sir0, sar0, popt0 = bss_eval_sources(src_ref, src_anchor)
    avg_SDRi = ((sdr[0]-sdr0[0]) + (sdr[1]-sdr0[1])) / 2
    # print("SDRi1: {0:.2f}, SDRi2: {1:.2f}".format(sdr[0]-sdr0[0], sdr[1]-sdr0[1]))
    return avg_SDRi


def cal_SISNRi(src_ref, src_est, mix):
    """Calculate Scale-Invariant Source-to-Noise Ratio improvement (SI-SNRi)
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T]
    Returns:
        average_SISNRi
    """
    sisnr1 = cal_SISNR(src_ref[0], src_est[0])
    sisnr2 = cal_SISNR(src_ref[1], src_est[1])
    sisnr1b = cal_SISNR(src_ref[0], mix)
    sisnr2b = cal_SISNR(src_ref[1], mix)
    avg_SISNRi = ((sisnr1 - sisnr1b) + (sisnr2 - sisnr2b)) / 2
    return avg_SISNRi


def cal_SISNR(ref_sig, out_sig, eps=1e-8):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        ref_sig: numpy.ndarray, [T]
        out_sig: numpy.ndarray, [T]
    Returns:
        SISNR
    """
    assert len(ref_sig) == len(out_sig)
    ref_sig = ref_sig - np.mean(ref_sig)
    out_sig = out_sig - np.mean(out_sig)
    ref_energy = np.sum(ref_sig ** 2) + eps
    proj = np.sum(ref_sig * out_sig) * ref_sig / ref_energy
    noise = out_sig - proj
    ratio = np.sum(proj ** 2) / (np.sum(noise ** 2) + eps)
    sisnr = 10 * np.log(ratio + eps) / np.log(10.0)
    return sisnr

def remove_pad(inputs, inputs_lengths):
    """
    Args:
        inputs: torch.Tensor, [B, C, T] or [B, T], B is batch size
        inputs_lengths: torch.Tensor, [B]
    Returns:
        results: a list containing B items, each item is [C, T], T varies
    """
    results = []
    dim = inputs.ndim
    if dim == 3:
        C = inputs.shape[1]
    # for input, length in zip(inputs, inputs_lengths):
    #     if dim == 3:  # [B, C, T]
    #         results.append(input[:, :length].view(C, -1).asnumpy())
    #     elif dim == 2:  # [B, T]
    #         results.append(input[:length].view(-1).asnumpy())
    for i, input in enumerate(inputs):
        if dim == 3:    # [B, C, T]
            results.append(input[:, :inputs_lengths[i]].view(C, -1).asnumpy())
        elif dim == 2:  # [B, T]
            results.append(input[:inputs_lengths[i]].view(-1).asnumpy())
    return results

if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=6)
    args = parser.parse_args()
    print(args)
    evaluate(args)
