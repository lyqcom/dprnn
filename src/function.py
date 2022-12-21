
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

""" A function to resolve the adaptability of the Graph schema """

import json
import os
import librosa
import numpy as np
import argparse
from mindspore import context

parser = argparse.ArgumentParser('function list')
parser.add_argument('--data_dir', type=str,
                    default=r"/home/heu_MEDAI/zhangyu/project/out_dir/tt",
                    help='directory including mix.json, s1.json and s2.json')
def get_input_with_list(dir):
    """ return data as a list """
    sample_rate = 8000
    L = 40
    mix_json = os.path.join(dir, 'mix.json')
    with open(mix_json, 'r') as f:
        mix_infos = json.load(f)
    input_with_list = []
    for mix_info in mix_infos:
        mix_path = mix_info[0]
        mix, _ = librosa.load(mix_path, sr=sample_rate)
        K = int(np.ceil(len(mix) / L))
        # print(K)
        input_with_list.append(K)
        input_with_list.sort(reverse=True)
    return input_with_list

if __name__ == '__main__':
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend", device_id=6)
    args = parser.parse_args()
    list = get_input_with_list(args.data_dir)
    print(list)