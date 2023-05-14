# %%
import os, sys
import argparse
import logging, json
import pandas as pd

import HD
import dataset_utils

import numpy as np

import torch



# %%
parser = argparse.ArgumentParser()
parser.add_argument("-algorithm", type=str, required=True,
                    help="Algorithm")
parser.add_argument("-dataset", type=str, default='mnist',
                    choices=['isolet', 'mnist', 'face', 'cardio3', 'pamap2', 'ucihar'],
                    help="Dataset selection")
parser.add_argument("-batch_size", type=int, default=256,
                    help="Batch size for metric training")
parser.add_argument("-D", type=int, default=1024,
                    help="HDC dimension")

parser.add_argument("-R", type=float, nargs='+', default=[2500, 16000],
                    help="R values")
parser.add_argument("-R_deviation", type=float, nargs='+', default=[0.18, 0.45],
                    help="R deviation")
parser.add_argument("-S_ou", type=int, default=8,
                    help="Number of activated WL in ReRAM")

parser.add_argument("-model_path", type=str, required=True,
                    help="Model checkpoint path")
parser.add_argument("-class_path", type=str, required=True,
                    help="Class HV path")
parser.add_argument("-log_path", type=str, required=True,
                    help="Logging path")

args = parser.parse_args()

dataset = args.dataset # 'mnist'
# cardio3 is not stable
batch_size = args.batch_size #256
D = args.D # 1024


kargs_hd_rram_test = {
    "S_ou": args.S_ou,
    "R": args.R,
    "R_deviation": args.R_deviation,
}


log_filename = '{}_{}_{}_{}_{}_{}'.format(args.algorithm, dataset, D, args.R, args.R_deviation, args.S_ou)

logging.basicConfig(filename=os.path.join(args.log_path, log_filename+'.log'),
            format='%(asctime)s %(message)s',
            filemode='w')
logger=logging.getLogger()
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.DEBUG)
logger.info(vars(args))

device = 'cuda'


# Load datasets
nFeatures, nClasses, x_train, y_train, x_test, y_test, train_loader, test_loader = dataset_utils.load_dataset(dataset, batch_size, device)
logger.info("Loaded dataset: {} with {} features, {} classes, train size={}, test size={}".format(dataset, nFeatures, nClasses, len(x_train), len(x_test)))

model = HD.HDC(dim=nFeatures, D=D, num_classes=nClasses, binary=True)
# model.init_class(x_train, y_train)

# Load model from ckpt
model_path, class_path = args.model_path, args.class_path
model.load_state_dict(torch.load(model_path), strict=False)

model.class_hvs.data = torch.load(class_path)
logger.info("Loaded model from: {} class from: {}".format(model_path, class_path))


configs = vars(args)
all_results = {
    'train_type': [], 'epoch': [],
    'fp_acc': [],
    'reram_acc': [],
    'ckpt_filename': []}

def update_result_dict(
    train_type, epoch_i, reram_test_dict, ckpt_filename):
    all_results['train_type'].append(train_type)
    all_results['epoch'].append(epoch_i)
    all_results['fp_acc'].append(reram_test_dict['fp_acc'])
    all_results['reram_acc'].append(reram_test_dict['test_acc'])
    all_results['ckpt_filename'].append(ckpt_filename)


res_rram = HD.test_RRAM_HD(model, x_test, y_test, kargs_hd_rram_test)
logger.info(res_rram)

update_result_dict('eval', -1, res_rram, '')

with open(os.path.join(args.log_path, log_filename + '.dat'), 'w') as convert_file:
    convert_file.write(json.dumps(configs))

df = pd.DataFrame(all_results)
df.to_csv(os.path.join(args.log_path, log_filename + '.csv'), index=False)
