"""
    获取对数据集中数据的采样，并以json文件格式保存
"""
import os, sys

sys.path.append(os.getcwd())
from util.utils import Sampler
import json
from tqdm import tqdm

max_step = 5
seg_len = 128
n_speaker = 2
mel_band = 80
lin_band = 513
n_samples = 300000
dset = 'train'

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('usage: python3 make_single_samples.py [in_h5py_path] [out_json_path]')
        exit(0)
    sampler = Sampler(sys.argv[1], max_step=max_step, seg_len=seg_len, dset=dset, n_speaker=n_speaker)
    samples = [sampler.sample_single()._asdict() for _ in tqdm(range(n_samples))]
    with open(sys.argv[2], 'w') as f_json:
        json.dump(samples, f_json, indent=4, separators=(',', ': '))
