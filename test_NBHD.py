#%%
from cProfile import label
import sys

import NBHD
import dataset_utils
import HD

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pytorch_metric_learning import distances, losses, miners, reducers
import matplotlib.pyplot as plt




#not sure if correct, get kinda weird distances
def get_Cosine_margin(model, x_test, y_test=None):
    def cosine_distance(a, b):
        return 1 - torch.cosine_similarity(a[:, np.newaxis, :], b, dim=-1)
    
    # Compute test samples' Hamming distance
    class_hvs = model.class_hvs.data
    test_enc_hvs = model(x_test, True)
    test_Cosine_dist = cosine_distance(test_enc_hvs, class_hvs)

    sorted_test_Cosine_distance, _ = torch.sort(test_Cosine_dist, dim=-1, descending=False)
    test_enc_hvs_Cosine_margin = (sorted_test_Cosine_distance[:,1:]-sorted_test_Cosine_distance[:,0].unsqueeze(dim=1)).mean(dim=1).cpu()
    mean_test_Cosine_margin = torch.mean(test_enc_hvs_Cosine_margin).item()

    return mean_test_Cosine_margin

def HD_test(model, x_test, y_test):
	out = model(x_test, embedding=False)
	preds = torch.argmax(out, dim=-1)

	acc = torch.mean((preds==y_test).float())	
	return acc



def test(dataset, model_type="NB", D = 1024, levels = 3):
    nFeatures, nClasses, x_train, y_train, x_test, y_test, train_loader, test_loader\
    = dataset_utils.load_dataset(dataset, 256, "cpu")


    if(model_type == "NB"):
        model = NBHD.HDC(dim = nFeatures, D=D, 
            num_classes=nClasses, levels=levels)
    elif(model_type == "B"):
        model = HD.HDC(device="cpu",dim=nFeatures, D=D, 
            num_classes=nClasses, binary=True)
    else:
        raise Exception("unknown model type")

    model.init_class(x_train, y_train)
    accuracies = []
    accuracies.append(HD_test(model, x_test, y_test))
    margins = []
    margins.append(get_Cosine_margin(model, x_test, y_test))
    
    num_HD_epoch = 10

    for epoch_i in range(1, num_HD_epoch+1):
        model.HD_train_step(x_train, y_train)
        accuracies.append(HD_test(model, x_test, y_test))
        margins.append(get_Cosine_margin(model, x_test, y_test))
        

    device = "cpu"
    num_metric_epochs = 0
    metric_lr = 0.001
    model.class_hvs = nn.parameter.Parameter(data=model.class_hvs)
    optimizer = optim.Adam(model.parameters(), lr=metric_lr)
    distance = distances.CosineSimilarity()

    reducer = reducers.ThresholdReducer(low=0.0)
    loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)
    mining_func = miners.TripletMarginMiner(
        margin=0.2, distance=distance, type_of_triplets="semihard", )

    ### pytorch-metric-learning stuff ###
    for epoch_i in range(1, num_metric_epochs + 1):
        NBHD.metric_train(model, loss_func, mining_func, device, train_loader, optimizer, epoch_i)
        accuracies.append(HD_test(model, x_test, y_test))
        margins.append(get_Cosine_margin(model, x_test, y_test))

    return accuracies, margins

#%%
#stability 

datasets = ["pamap2", "isolet", "cardio3", "ucihar"]

for dataset in datasets:
    plt.title(dataset)

    NB_accuracies, NB_margins = test(dataset, model_type="NB")
    plt.plot(NB_accuracies, label="Non-Binary Accuracy", c = "red")
    plt.plot(NB_margins, label="Non-Binary Margins", c = "orange")

    plt.axvline(x = 10, ls="--")

    B_accuracies, B_margins = test(dataset, model_type="B")
    plt.plot(B_accuracies, label="Binary Accuracy", c = "blue")
    plt.plot(B_margins, label="Binary Margins", c="cyan")

    plt.xlabel("iterations")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()
# %%
#overall accuracy
datasets = ["pamap2", "isolet", "cardio3", "ucihar"]
dimensions = [512, 1024, 2048, 4096]
for dataset in datasets:
    plt.title(dataset)
    NB_accuracies = []
    B_accuracies = []
    for dimension in dimensions: 
        NB_acc, NB_margins = test(dataset, D=dimension, model_type="NB")
        NB_accuracies.append(max(NB_acc))

        B_acc, B_margins = test(dataset, D=dimension,  model_type="B")
        B_accuracies.append(max(B_acc))


    plt.plot(dimensions, NB_accuracies, label="Non-Binary")
    plt.plot(dimensions, B_accuracies, label="Binary")
    plt.xlabel("dimensions")
    plt.ylabel("Maximum Accuracies")
    plt.legend()
    plt.show()
# %%
