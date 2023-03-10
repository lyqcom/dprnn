# Created on 2018/12
# Author: Kaituo XU
"""
Logic:
1. AudioDataLoader generate a minibatch from AudioDataset, the size of this
   minibatch is AudioDataLoader's batchsize. For now, we always set
   AudioDataLoader's batchsize as 1. The real minibatch size we care about is
   set in AudioDataset's __init__(...). So actually, we generate the
   information of one minibatch in AudioDataset.
2. After AudioDataLoader getting one minibatch from AudioDataset,
   AudioDataLoader calls its collate_fn(batch) to process this minibatch.

Input:
    Mixtured WJS0 tr, cv and tt path
Output:
    One batch at a time.
    Each inputs's shape is B x T
    Each targets's shape is B x C x T
"""

import json
import math
import os
import librosa
import argparse

import numpy as np

import mindspore.dataset as ds
from mindspore import context


parser = argparse.ArgumentParser(
    "Dual-path transformer"
    "with Permutation Invariant Training")
parser.add_argument('--train_dir', type=str, default='/home/zhangyu/DPRNNmodel618-ms15/out_dir/tr',
                    help='directory including mix.json, s1.json and s2.json')
# parser.add_argument('--valid_dir', type=str, default='/home/heu_MEDAI/zhangyu/project/out_dir/cv',
#                     help='directory including mix.json, s1.json and s2.json')
# parser.add_argument('--train_dir', type=str, default='/mass_data/dataset/LS-2mix/Libri2Mix/tr',
#                     help='directory including mix.json, s1.json and s2.json')
# parser.add_argument('--valid_dir', type=str, default='/mass_data/dataset/LS-2mix/Libri2Mix/cv',
#                     help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--batch_size', default=3, type=int,   #default =3
                    help='Batch size')
parser.add_argument('--sample_rate', default=8000, type=int,
                    help='Sample rate')
parser.add_argument('--segment', default=4, type=float,
                    help='Segment length (seconds)')

def load_mixtures_and_sources(batch):
    """
    Each info include wav path and wav duration.
    Returns:
        mixtures: a list containing B items, each item is T np.ndarray
        sources: a list containing B items, each item is T x C np.ndarray
        T varies from item to item.
    """
    mixtures, sources = [], []
    mix_infos, s1_infos, s2_infos, sample_rate, segment_len = batch
    # for each utterance
    for mix_info, s1_info, s2_info in zip(mix_infos, s1_infos, s2_infos):
        mix_path = mix_info[0]
        s1_path = s1_info[0]
        s2_path = s2_info[0]
        assert mix_info[1] == s1_info[1] and s1_info[1] == s2_info[1]
        # read wav file
        mix, _ = librosa.load(mix_path, sr=sample_rate)
        s1, _ = librosa.load(s1_path, sr=sample_rate)
        s2, _ = librosa.load(s2_path, sr=sample_rate)
        # merge s1 and s2
        s = np.dstack((s1, s2))[0]  # T x C, C = 2
        utt_len = mix.shape[-1]
        if segment_len >= 0:
            # segment
            for i in range(0, utt_len - segment_len + 1, segment_len):
                mixtures.append(mix[i:i+segment_len])
                sources.append(s[i:i+segment_len])
            if utt_len % segment_len != 0:
                mixtures.append(mix[-segment_len:])
                sources.append(s[-segment_len:])
        else:  # full utterance
            mixtures.append(mix)
            sources.append(s)
    return mixtures, sources


def pad_list(xs):
    n_batch = len(xs)
    max_len = max(x.shape for x in xs)
    if len(max_len) == 1:
        pad = np.zeros((n_batch, max_len[0]), np.float32)
    else:
        pad = np.zeros((n_batch, max_len[0], max_len[1]), np.float32)
    for i in range(n_batch):
        temp = xs[i].shape
        pad[i, :temp[0]] = xs[i]
    return pad


class DatasetGenerator:

    def __init__(self, json_dir, batch_size, sample_rate=8000, segment=4.0, cv_maxlen=8.0):
        """
        Args:
            json_dir: directory including mix.json, s1.json and s2.json
            segment: duration of audio segment, when set to -1, use full audio

        xxx_infos is a list and each item is a tuple (wav_file, #samples)
        """
        super(DatasetGenerator, self).__init__()
        mix_json = os.path.join(json_dir, 'mix.json')
        s1_json = os.path.join(json_dir, 's1.json')
        s2_json = os.path.join(json_dir, 's2.json')
        with open(mix_json, 'r') as f:
            mix_infos = json.load(f)
        with open(s1_json, 'r') as f:
            s1_infos = json.load(f)
        with open(s2_json, 'r') as f:
            s2_infos = json.load(f)
        # sort it by #samples (impl bucket)
        def sort(infos): return sorted(
            infos, key=lambda info: int(info[1]), reverse=True)
        sorted_mix_infos = sort(mix_infos)
        sorted_s1_infos = sort(s1_infos)
        sorted_s2_infos = sort(s2_infos)
        if segment >= 0.0:
            # segment length and count dropped utts
            segment_len = int(segment * sample_rate)  # 4s * 8000/s = 32000 samples
            drop_utt, drop_len = 0, 0
            for _, sample in sorted_mix_infos:
                if sample < segment_len:
                    drop_utt += 1
                    drop_len += sample
            print("Drop {} utts({:.2f} h) which is short than {} samples".format(
                drop_utt, drop_len/sample_rate/36000, segment_len))
            # generate minibach infomations
            mixture_pad =[]
            lens = []
            source_pad = []
            start = 0
            while True:
                num_segments = 0
                end = start
                part_mix, part_s1, part_s2 = [], [], []
                while num_segments < batch_size and end < len(sorted_mix_infos):
                    utt_len = int(sorted_mix_infos[end][1])
                    if utt_len >= segment_len:  # skip too short utt
                        num_segments += math.ceil(utt_len / segment_len)
                        # Ensure num_segments is less than batch_size
                        if num_segments > batch_size:
                            # if num_segments of 1st audio > batch_size, skip it
                            if start == end: end += 1
                            break
                        part_mix.append(sorted_mix_infos[end])
                        part_s1.append(sorted_s1_infos[end])
                        part_s2.append(sorted_s2_infos[end])
                    end += 1
                if len(part_mix) > 0:
                    meta = [part_mix, part_s1, part_s2, sample_rate, segment_len]
                    mixtures_pad, ilens, sources_pad = self.sort_and_pad(meta)
                    for i in range(len(mixtures_pad)):
                        mixture_pad.append(mixtures_pad[i])
                        lens.append(ilens[i])
                        source_pad.append(sources_pad[i])
                if end == len(sorted_mix_infos):
                    break
                start = end
            self.mixture = mixture_pad
            self.len = lens
            self.sources = source_pad
        else:  # Load full utterance but not segment
            # generate minibach infomations
            mixture_pad = []
            lens = []
            source_pad = []
            start = 0
            while True:
                end = min(len(sorted_mix_infos), start + batch_size)
                # Skip long audio to avoid out-of-memory issue
                if int(sorted_mix_infos[start][1]) > cv_maxlen * sample_rate:
                    start = end
                    continue
                meta = [sorted_mix_infos[start:end], sorted_s1_infos[start:end], sorted_s2_infos[start:end], sample_rate, segment]
                mixtures_pad, ilens, sources_pad = self.sort_and_pad(meta)
                for i in range(len(mixtures_pad)):
                    mixture_pad.append(mixtures_pad[i])
                    lens.append(ilens[i])
                    source_pad.append(sources_pad[i])
                if end == len(sorted_mix_infos):
                    break
                start = end
            self.mixture = mixture_pad
            self.len = lens
            self.sources = source_pad

    def __getitem__(self, index):
        return (self.mixture[index], self.len[index], self.sources[index])

    def __len__(self):
        return len(self.mixture)


    def sort_and_pad(self, batch):
        #assert len(batch) == 1
        mixtures, sources = load_mixtures_and_sources(batch)

        # get batch of lengths of input sequences
        ilens = np.array([mix.shape[0] for mix in mixtures])

        mixtures_pad = pad_list([mix for mix in mixtures])

        sources_pad = pad_list([s for s in sources])

        sources_pad = sources_pad.transpose((0, 2, 1))
        return mixtures_pad, ilens, sources_pad



if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=1)
    args = parser.parse_args()
    print(args)
    tr_dataset = DatasetGenerator(args.train_dir, args.batch_size,
                              sample_rate=args.sample_rate, segment=args.segment)
    dataset = ds.GeneratorDataset(tr_dataset, ["mixture", "lens", "sources"], shuffle=False)
    dataset = dataset.batch(batch_size=2)
    iter_per_epoch = dataset.get_dataset_size()
    print(iter_per_epoch)
    i=0
    for data in dataset.create_dict_iterator():
        #print(data["mixture"], data["lens"], data["sources"])
        i+=1
        print(i)
        print(data["mixture"])
        print(data["mixture"].shape)
        print(data["lens"])
        print(data["lens"].shape)
        print(data["sources"])
        print(data["sources"].shape)
