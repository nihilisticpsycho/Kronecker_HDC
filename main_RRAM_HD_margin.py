# %%
import os, sys
import argparse
import logging, json
import pandas as pd

import HD
import dataset_utils

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from pytorch_metric_learning import distances, losses, miners, reducers


# %%
parser = argparse.ArgumentParser()
parser.add_argument("-dataset", type=str, default='mnist',
                    choices=['isolet', 'mnist', 'face', 'cardio3', 'pamap2', 'ucihar'],
                    help="Dataset selection")
parser.add_argument("-batch_size", type=int, default=256,
                    help="Batch size for metric training")
parser.add_argument("-D", type=int, default=1024,
                    help="HDC dimension")
parser.add_argument("-epoch_HDC", type=int, default=1,
                    help="Number of epochs for HDC training")
parser.add_argument("-epoch_metric", type=int, default=3,
                    help="Number of epochs for metric training")
parser.add_argument("-lr", type=float, default=0.001,
                    help="Number of epochs for metric training")

parser.add_argument("-S_ou", type=int, default=8,
                    help="Number of activated WL in ReRAM")

parser.add_argument("-log_path", type=str, default='./log',
                    help="Logging path")

args = parser.parse_args()

dataset = args.dataset # 'mnist'
# cardio3 is not stable
batch_size = args.batch_size #256
D = args.D # 1024

num_HD_epoch = args.epoch_HDC # 3
num_metric_epochs = args.epoch_metric #15
metric_lr = args.lr # 0.1


kargs_hd_rram_test = {
    "S_ou": args.S_ou,
    "R": [2500, 16000],
    "R_deviation": [0.18, 0.45],
}


log_path = args.log_path
log_filename = os.path.join(log_path, "log.log") if log_path else "log.log"

logging.basicConfig(filename=log_filename,
            format='%(asctime)s %(message)s',
            filemode='w')
logger=logging.getLogger()
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

logger.setLevel(logging.DEBUG)

device = "cuda"

logger.info(vars(args))

# Load datasets
nFeatures, nClasses, x_train, y_train, x_test, y_test, train_loader, test_loader = dataset_utils.load_dataset(dataset, batch_size, device)
logger.info("Loaded dataset: {} with {} features, {} classes, train size={}, test size={}".format(dataset, nFeatures, nClasses, len(x_train), len(x_test)))

model = HD.HDC(dim=nFeatures, D=D, num_classes=nClasses, binary=True)
model.init_class(x_train, y_train)

configs = vars(args)
all_results = {
    'train_type': [], 'epoch': [],
    'avg_class_Hamming_margin': [],
    'avg_test_Hamming_margin': [],
    'fp_acc': [],
    'reram_acc': [],
    'ckpt_filename': []}

def update_result_dict(
    train_type, epoch_i,
    dict_Hamming_margin, reram_test_dict, ckpt_filename):
    all_results['train_type'].append(train_type)
    all_results['epoch'].append(epoch_i)
    all_results['avg_class_Hamming_margin'].append(dict_Hamming_margin['avg_class_Hamming_dist'])
    all_results['avg_test_Hamming_margin'].append(dict_Hamming_margin['avg_test_Hamming_margin'])
    all_results['fp_acc'].append(reram_test_dict['fp_acc'])
    all_results['reram_acc'].append(reram_test_dict['test_acc'])
    all_results['ckpt_filename'].append(ckpt_filename)


# 1. HD Training 
for epoch_i in range(1, num_HD_epoch+1):
    logger.info("\n=====================================\nHD training epoch: {}".format(epoch_i))
    model.HD_train_step(x_train, y_train)

    res_Hamming = HD.get_Hamming_margin(model, x_test)
    logger.info(res_Hamming)
    res_rram = HD.test_RRAM_HD(model, x_test, y_test, kargs_hd_rram_test)
    logger.info(res_rram)

    MODEL_FILENAME = 'model_hdc_epoch_{}.ckpt'.format(epoch_i)
    torch.save(model.state_dict(), os.path.join(log_path, MODEL_FILENAME))

    CLASS_FILENAME = 'model_hdc_class_hvs_{}.pt'.format(epoch_i)
    torch.save(model.class_hvs, os.path.join(log_path, CLASS_FILENAME))

    update_result_dict('hdc', epoch_i, res_Hamming, res_rram, MODEL_FILENAME)

model.class_hvs = nn.parameter.Parameter(data=model.class_hvs)
optimizer = optim.Adam(model.parameters(), lr=metric_lr)
# optimizer = optim.SGD(model.parameters(), lr=metric_lr)

### pytorch-metric-learning stuff ###
distance = distances.CosineSimilarity()

reducer = reducers.ThresholdReducer(low=0.0)
loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)
mining_func = miners.TripletMarginMiner(
    margin=0.2, distance=distance, type_of_triplets="semihard", )

### pytorch-metric-learning stuff ###
for epoch_i in range(1, num_metric_epochs + 1):
    logger.info("\n=====================================\nMetric learning epoch: {}".format(epoch_i))
    HD.metric_train(model, loss_func, mining_func, device, train_loader, optimizer, epoch_i)
    
    res_Hamming = HD.get_Hamming_margin(model, x_test)
    logger.info(res_Hamming)
    res_rram = HD.test_RRAM_HD(model, x_test, y_test, kargs_hd_rram_test)
    logger.info(res_rram)

    MODEL_FILENAME = 'model_metric_epoch_{}.ckpt'.format(epoch_i)
    torch.save(model.state_dict(), os.path.join(log_path, MODEL_FILENAME))

    CLASS_FILENAME = 'model_metric_class_hvs_{}.pt'.format(epoch_i)
    torch.save(model.class_hvs, os.path.join(log_path, CLASS_FILENAME))

    update_result_dict('metric', epoch_i, res_Hamming, res_rram, MODEL_FILENAME)



with open(os.path.join(log_path, 'configs.dat'), 'w') as convert_file:
    convert_file.write(json.dumps(configs))

df = pd.DataFrame(all_results)
df.to_csv(os.path.join(log_path, 'results.csv'), index=False)
