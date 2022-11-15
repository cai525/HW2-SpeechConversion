"""对单个语音进行转换"""
from itertools import permutations
import os
import re

from scipy.io import wavfile

from preprocess.tacotron.norm_utils import get_spectrograms
from convert import convert_sp, get_model, sp2wav


def convert_single(src_file_path,sp_id,solver):
    """ 单语者转化

    Args:
        src_file_path: 原语音目录
        sp_id: 输出语音的说话人（即目标转化者）的 id

    Returns:
        wave_data: 完成转化的语音

    """
    _, lin_spec = get_spectrograms(src_file_path)  # 读取音频并抽取特征
    converted_sp = convert_sp(lin_spec, sp_id, solver, gen=True)  # 语音转换
    wav_data = sp2wav(converted_sp)
    return wav_data


if __name__ == '__main__':
    # 音频参数
    audio_dir = './archive/data/wav_test'  # 音频目录
    speakers = ['p1', 'p2']  # 语者
    speakers_id = {speaker: i for i, speaker in enumerate(speakers)}  # 语者id:应与训练所用id对应
    res_dir = './archive/results'  # 保存结果目录
    fs = 16000  # 采样率
    # 模型参数
    model_path = './archive/model_weight/final/model.pkl-79000'  # 模型权重路径
    hps_path = './archive/hps/vctk.json'  # 模型超参数设置储存路径
    solver = get_model(hps_path=hps_path, model_path=model_path)
    for src_speaker, tar_speaker in permutations(speakers, 2):
        src_names = os.listdir(os.path.join(audio_dir, src_speaker).replace('\\', '/'))  # src_name 格式范例:p1_366
        for src_file_name in src_names:
            index = re.search(r'[0-9]{3}', src_file_name).group(0)
            src_file_path = os.path.join(audio_dir, src_speaker, src_file_name).replace('\\', '/')
            wav_data = convert_single(src_file_path,speakers_id[tar_speaker],solver)
            wav_path = os.path.join(res_dir, src_speaker + '_' + tar_speaker,
                                    src_speaker + '_' + tar_speaker + '_' + index + '.wav').replace('\\', '/')
            wavfile.write(wav_path, fs, wav_data)

