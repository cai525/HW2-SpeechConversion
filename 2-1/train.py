from util.utils import Hps
from util.utils import DataLoader
from util.utils import SingleDataset
from src.solver import Solver
import argparse


def get_argument():
	"""获取命令行参数"""
	parser = argparse.ArgumentParser()
	parser.add_argument('--train', default=True, action='store_true')
	parser.add_argument('--test', default=False, action='store_true')
	parser.add_argument('--load_model', default=False, action='store_true')
	parser.add_argument('-flag', default='train')
	parser.add_argument('-hps_path', default='./archive/hps/vctk.json')
	parser.add_argument('-load_model_path', default='')
	parser.add_argument('-dataset_path', default='./archive/data/wav48/dataset_p1_p2.h5')
	parser.add_argument('-index_path', default='./archive/data/wav48/index_p1_p2.json')
	parser.add_argument('-output_model_path', default='./archive/model_weight/vc')
	parser.add_argument('-last_iter',default='0')
	return parser.parse_args()


if __name__ == '__main__':
	args = get_argument()
	hps = Hps()
	hps.load(args.hps_path)
	hps_tuple = hps.get_tuple()
	dataset = SingleDataset(args.dataset_path,
							args.index_path,
							seg_len=hps_tuple.seg_len)

	data_loader = DataLoader(dataset)

	solver = Solver(hps_tuple, data_loader)
	if args.load_model:
		solver.load_model(args.load_model_path)
		solver.last_iter = int(args.last_iter)
	if args.train:
		solver.train(args.output_model_path, args.flag, mode='pretrain_G')
		solver.save_model('./archive/model_weight/vc-generator', enc_only=False)

		solver.train(args.output_model_path, args.flag, mode='pretrain_D')
		solver.save_model('./archive/model_weight/vc-discriminator', enc_only=False)
		solver.train(args.output_model_path, args.flag, mode='train')
		solver.train(args.output_model_path, args.flag, mode='patchGAN')
