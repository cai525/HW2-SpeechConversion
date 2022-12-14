"""
    将数据划分为训练集和测试集，并将数据保存为.h5格式文件

"""
import h5py
import numpy as np
import sys
import os
import glob
import re
from collections import defaultdict
from tacotron.norm_utils import get_spectrograms
# ============= 修改此处以符合自己的配置 ================
root_dir = '../archive/data/wav48'       # 数据集根路径
speaker_used_list = np.loadtxt('../archive/hps/en_speaker_used.txt')
train_split = 0.9
# =================================================


def read_speaker_info(path='../archive/data/speaker-info.txt'):
    accent2speaker = defaultdict(lambda: [])
    with open(path) as f:
        splited_lines = [line.strip().split() for line in f][1:]
        speakers = [line[0] for line in splited_lines]
        accents = [line[3] for line in splited_lines]
        for speaker, accent in zip(speakers, accents):
            accent2speaker[accent].append(speaker)
    return accent2speaker


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: python3 make_dataset_vctk.py [h5py_path]')
        exit(0)
    accent2speaker = read_speaker_info()
    h5py_path = sys.argv[1]
    filename_groups = defaultdict(lambda: [])
    with h5py.File(h5py_path, 'w') as f_h5:
        filenames = sorted(glob.glob(os.path.join(root_dir, '*/*.wav')))
        for filename in filenames:
            filename = filename.replace('\\','/')
            # divide into groups
            sub_filename = filename.strip().split('/')[-1]
            # format: p{speaker}_{sid}.wav
            speaker_id, utt_id = re.search(r'p(\d+)_(\d+)\.wav', sub_filename).groups()
            filename_groups[speaker_id].append(filename)
        for speaker_id, filenames in filename_groups.items():
            if speaker_id not in accent2speaker['English']:
                continue
            if int(speaker_id) not in speaker_used_list:
                continue
            print('processing {}'.format(speaker_id))
            train_size = int(len(filenames) * train_split)
            for i, filename in enumerate(filenames):
                print(filename)
                sub_filename = filename.strip().split('/')[-1]
                # format: p{speaker}_{sid}.wav
                speaker_id, utt_id = re.search(r'p(\d+)_(\d+)\.wav', sub_filename).groups()
                mel_spec, lin_spec = get_spectrograms(filename)
                if i < train_size:
                    datatype = 'train'
                else:
                    datatype = 'test'
                f_h5.create_dataset(f'{datatype}/{speaker_id}/{utt_id}/mel',
                                    data=mel_spec, dtype=np.float32)
                f_h5.create_dataset(f'{datatype}/{speaker_id}/{utt_id}/lin',
                                    data=lin_spec, dtype=np.float32)
