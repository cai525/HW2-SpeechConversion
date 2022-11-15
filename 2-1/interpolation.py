import numpy as np
import torch.nn.functional as F
import torch

from convert import sp2wav
from preprocess.tacotron.norm_utils import get_spectrograms
from src.model import Decoder, RNN, linear, append_emb
from src.solver import Solver
from util.utils import Hps


class Decoder_with_interpolation(Decoder):
    def __init__(self, rate=0, c_in=512, c_out=513, c_h=512, c_a=2, emb_size=512, ns=0.01):
        """
        Args:
            rate: 插值率,p2 的声音所占的比率
            c_in, c_out,c_h, c_a, emb_size, ns: 见父类
        """
        super(Decoder_with_interpolation, self).__init__(c_in, c_out, c_h, c_a, emb_size, ns)
        self.rate = rate

    def forward(self, x, c):
        f_mix = lambda a, b: (self.rate * a + (1 - self.rate) * b).view(1,-1)  # 嵌入层融合函数
        # 融合层
        p1 = torch.tensor(0,dtype=torch.int).to("cuda")
        p2 = torch.tensor(1,dtype=torch.int).to("cuda")
        emb1 = f_mix(self.emb1(p1), self.emb1(p2))
        emb2 = f_mix(self.emb2(p1), self.emb2(p2))
        emb3 = f_mix(self.emb3(p1), self.emb3(p2))
        emb4 = f_mix(self.emb4(p1), self.emb4(p2))
        emb5 = f_mix(self.emb5(p1), self.emb5(p2))

        out = self.conv_block(x, [self.conv1, self.conv2], self.ins_norm1, emb1, res=True)
        out = self.conv_block(out, [self.conv3, self.conv4], self.ins_norm2, emb2, res=True)
        out = self.conv_block(out, [self.conv5, self.conv6], self.ins_norm3, emb3, res=True)
        # dense layer
        out = self.dense_block(out, emb4, [self.dense1, self.dense2], self.ins_norm4, res=True)
        out = self.dense_block(out, emb4, [self.dense3, self.dense4], self.ins_norm5, res=True)
        emb = emb5
        out_add = out + emb.view(emb.size(0), emb.size(1), 1)
        # rnn layer
        out_rnn = RNN(out_add, self.RNN)
        out = torch.cat([out, out_rnn], dim=1)
        out = append_emb(emb5, out.size(2), out)
        out = linear(out, self.dense5)
        out = F.leaky_relu(out, negative_slope=self.ns)
        out = linear(out, self.linear)
        return out


def interpolation_convert(rate, src_file_path, model_path, hps_path):
    # 导入模型
    hps = Hps()
    hps.load(hps_path)
    hps_tuple = hps.get_tuple()
    solver = Solver(hps_tuple, None, None)
    solver.load_model(model_path)

    # 加载solver
    print('load model from {}'.format(model_path))
    with open(model_path, 'rb') as f_in:
        all_model = torch.load(f_in)
        solver.Encoder.load_state_dict(all_model['encoder'])
        # decoder采用新的结构
        new_decoder = Decoder_with_interpolation(rate=rate)
        new_decoder.load_state_dict(all_model['decoder'])
        solver.Decoder = new_decoder.cuda()
        # generator 也采用新结构
        new_gen = Decoder_with_interpolation(rate=rate)
        new_gen.load_state_dict(all_model['generator'])
        solver.Generator = new_gen.cuda()

    # 加载音频
    _, lin_spec = get_spectrograms(src_file_path)  # 读取音频并抽取特征
    sp_tensor = torch.from_numpy(np.expand_dims(lin_spec, axis=0))
    sp_tensor = sp_tensor.type(torch.FloatTensor)
    # 转化
    converted_sp = solver.test_step(sp_tensor, None, gen=True)
    converted_sp = converted_sp.squeeze(axis=0).transpose((1, 0))
    # 输出转wave
    wav_data = sp2wav(converted_sp)
    return wav_data


if __name__ == "__main__":
    model_path = './archive/model_weight/final/model.pkl-79000'  # 模型权重路径
    hps_path = './archive/hps/vctk.json'  # 模型超参数设置储存路径
    src_path = './archive/data/wav_test/p1/p1_331.wav'  # 音频目录
    rate = 0.5
    wav = interpolation_convert(rate,src_path,model_path,hps_path)
    print(wav)
