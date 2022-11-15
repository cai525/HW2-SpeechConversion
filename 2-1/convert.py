"""对数据集中划分的测试集进行转换"""
import os
import sys
import h5py
import torch
import numpy as np
from util.utils import Hps
from src.solver import Solver
from scipy.io import wavfile
from torch.autograd import Variable
from preprocess.tacotron.norm_utils import spectrogram2wav


def sp2wav(sp):
    """from spectrum to wave """
    exp_sp = sp
    wav_data = spectrogram2wav(exp_sp)
    return wav_data


def convert_sp(sp, c, solver, gen=True):
    c_var = Variable(torch.from_numpy(np.array([c]))).cuda()
    sp_tensor = torch.from_numpy(np.expand_dims(sp, axis=0))
    sp_tensor = sp_tensor.type(torch.FloatTensor)
    converted_sp = solver.test_step(sp_tensor, c_var, gen=gen)
    converted_sp = converted_sp.squeeze(axis=0).transpose((1, 0))
    return converted_sp


def get_model(hps_path, model_path):
    hps = Hps()
    hps.load(hps_path)
    hps_tuple = hps.get_tuple()
    solver = Solver(hps_tuple, None, None)
    solver.load_model(model_path)
    return solver


def convert_all_sp(h5_path, src_speaker, tar_speaker, solver, dir_path,
                   dset='test', gen=True, max_n=2,
                   speaker_used_path='./archive/hps/en_speaker_used.txt'):
    # read speaker id file
    with open(speaker_used_path) as f:
        speakers = [line.strip() for line in f]
        speaker2id = {speaker: i for i, speaker in enumerate(speakers)}

    with h5py.File(h5_path, 'r') as f_h5:
        c = 0
        for utt_id in f_h5[f'{dset}/{src_speaker}']:
            sp = f_h5[f'{dset}/{src_speaker}/{utt_id}/lin'][()]
            # convert speech
            converted_sp = convert_sp(sp, speaker2id[tar_speaker], solver, gen=gen)
            wav_data = sp2wav(converted_sp)
            wav_path = os.path.join(dir_path, f'{src_speaker}_{tar_speaker}_{utt_id}.wav')
            wavfile.write(wav_path, 16000, wav_data)
            c += 1
            if c >= max_n:
                break


if __name__ == '__main__':
    # ============= 根据需要修改 =====================
    h5_path = './archive/data/wav48/dataset_p1_p2.h5'           # 待转换的数据路径
    root_dir = './archive/results'                              # 存储结果的路径
    model_path = './archive/model_weight/final/vc_model.pkl'    # 模型权重路径
    hps_path = './archive/hps/vctk.json'                        # 模型超参数设置储存路径
    solver = get_model(hps_path=hps_path, model_path=model_path)
    speakers = ['1', '2']
    max_n = 2
    # ===============================================
    # if len(sys.argv) == 3:
    #     speakers = speakers[:min(2, int(sys.argv[1]))]
    #     max_n = min(2, int(sys.argv[2]))
    for speaker_A in speakers:
        for speaker_B in speakers:
            if speaker_A != speaker_B:
                dir_path = os.path.join(root_dir, f'p{speaker_A}_p{speaker_B}')
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                convert_all_sp(h5_path, speaker_A, speaker_B,
                               solver, dir_path, max_n=max_n)
