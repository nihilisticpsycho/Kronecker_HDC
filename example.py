import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd 
import torch as torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from pytorch_metric_learning import distances, losses, miners, reducers
from gkpd import gkpd, KroneckerConv2d
from gkpd.tensorops import kron

class BinaryModel(nn.Module):
    def __init__(self, dim, D, num_classes, enc_type='RP', binary=True, device='cpu', kargs=None):
        super(BinaryModel, self).__init__()
        self.enc_type, self.binary = enc_type, binary	
        self.device = device
        if enc_type in ['RP', 'RP-COS']:
            self.rp_layer = nn.Linear(dim, D).to(device)
            self.class_hvs = torch.zeros(num_classes, D).float().to(device)
            self.class_hvs_nb = torch.zeros(num_classes, D).float().to(device)
        else:
            pass
    #hard sigmoid    
    def weight_binarize(self, W):
       W = torch.where(W<-1,-1,W)
       W = torch.where(W>1,1,W)
       mask1 = (W >= -1) & (W < 0)
       W[mask1] = 2 * W[mask1] + W[mask1]*W[mask1]
       mask2 = (W >= 0) & (W < 1)
       W[mask2] = 2 * W[mask2] - W[mask2]*W[mask2]
       # W[W > 1] = 1
       return W
    #using Bi-Real Approximation     
    def activation_binarize(self,a):
       a = torch.where(a<-1,-1,a)
       a = torch.where(a>1,1,a)
       mask1 = (a >= -1) & (a < 0)
       a[mask1] = 2 * a[mask1] + a[mask1]*a[mask1]
       #a[mask1] = 0
       mask2 = (a >= 0) & (a < 1)
       a[mask2] = 2 * a[mask2] - a[mask2]*a[mask2]
       #a = torch.where((a >= -1) & (a < 0),2*a + torch.pow(a,2) )
       #a = torch.where((a >= 0) & (a < 1), 2*a- torch.pow(a,2))
    #    a [a < -1] = -1
    #    a [a > 1]   =  1
    #    a [(a >= -1) & (a < 0)] = 2*a[(a >= -1) & (a < 0)] + torch.pow(a [(a >= -1) & (a < 0)],2)
    #    a [(a >= 0) & (a < 1)] = 2*a[(a >= 0) & (a < 1)] - torch.pow(a [(a >= 0) & (a < 1)],2)
       return a

    def encoding(self, x):
        if self.enc_type == 'RP':
            #x = self.activation_binarize(x) 
            #need not binarize the inputs 
            #progressively binarize the inputs, after training the weights
            #add some print statements and check 
            #print("The value of weights, before binarization")
            #print(self.rp_layer.weight.data)
            #weights = self.rp_layer.weight.data.clone()
            #weights_bin = self.weight_binarize(weights)
            #self.rp_layer.weight.data = weights_bin.clone() 
            out = self.rp_layer(x)
            #print("The value of weights, after binarization")
            #print(self.rp_layer.weight.data)
        else:
                pass
        
        return binarize_soft(out) if self.binary else out
    
    #Forward Function
    def forward(self, x, embedding=False):
        out = self.encoding(x)
        if embedding:
            out = out
        else:
            out = self.similarity(class_hvs=binarize_hard(self.class_hvs), enc_hv=out)   
        return out
    
    def init_class(self, x_train, labels_train):
        out = self.encoding(x_train)
        if self.binary:
            out = binarize_hard(out)

        for i in range(x_train.size()[0]):
            self.class_hvs[labels_train[i]] += out[i]
        
        if self.binary:
            self.class_hvs = binarize_hard(self.class_hvs)
        
    def HD_train_step(self, x_train, y_train, lr=1.0):
        shuffle_idx = torch.randperm(x_train.size()[0])
        x_train = x_train[shuffle_idx]
        train_labels = y_train[shuffle_idx]
        l= list(self.rp_layer.parameters())
        enc_hvs = binarize_hard(self.encoding(x_train))
        for i in range(enc_hvs.size()[0]):
            sims = self.similarity(self.class_hvs, enc_hvs[i].unsqueeze(dim=0))
            predict = torch.argmax(sims, dim=1)
            
            if predict != train_labels[i]:
                self.class_hvs_nb[predict] -= lr * enc_hvs[i]
                self.class_hvs_nb[train_labels[i]] += lr * enc_hvs[i]
            
            self.class_hvs.data = binarize_hard(self.class_hvs_nb)
    
    def similarity(self, class_hvs, enc_hv):
	    return torch.matmul(enc_hv, class_hvs.t())/class_hvs.size()[1]
    

if __name__ == '__main__':
  # Reference code
   
  rank = 8
  filename = f'./model/binary_model_2000.pt'
  model = torch.load(filename)
  w = model.rp_layer.weight.data.view(2000,784,1,1)
  a_shape, b_shape = (rank, 40 , 28, 1, 1), (rank, 50, 28, 1, 1)
  
  # Full rank
  a, b  = torch.randn(*a_shape), torch.randn(*b_shape)
  print(*a_shape)
  #print(a)
  #w = kron(a,b)

  # Approximation
  a_hat, b_hat = gkpd(w, a_shape[1:], b_shape[1:])
  #print("a_b_hat  shape",a_b_hat.shape)
  #print("a_b_hat",a_b_hat)
  #w_hat = kron(a_b_hat, torch.ones(rank,1,1,1,1))
  w_hat = kron(a_hat, b_hat)
  # print("w_hat",a_b_hat)
  # w_hat_sum =torch.stack([a_b_hat[k] for k in range(a_b_hat.shape[0])]).sum(dim=0)
  # print("w_hat_sum",w_hat_sum)
  # for i in range(a_b_hat.shape[0]):
  #  #print(torch.abs(a_b_hat[i,:,:,:,:] - w_hat))
  #   print("a_b_hat",a_b_hat[i,:,:,:,:])
  #   print("w_hat",w_hat)
  #print("w_hat shape",w_hat.shape)
  #a_b_hat_t = torch.Tensor(a_b_hat.shape[1],a_b_hat.shape[2],a_b_hat.shape[3],a_b_hat.shape[4])
  #for i in range(1,a_b_hat.shape[0]):
  # a_b_hat_t = a_b_hat[0,:,:,:,:]
  # print("a_b_hat_t shape",a_b_hat_t.shape)
#   a_hat, b_hat =gkpd(w_hat_sum, a_shape[1:], b_shape[1:])
#   w_hat = kron(a_hat, b_hat)
#   #print("w_hat",w_hat.shape)
#   new_shape = (rank,w_hat.shape[0],w_hat.shape[1],w_hat.shape[2],w_hat.shape[3])
#   new_matrix = torch.Tensor(*new_shape)
#   for i in range(1, rank):
#     new_matrix[i,:, :, :,:] = w_hat
# #  new_matrix = torch.cat([torch.unsqueeze(w_hat,dim=0)]*rank,dim=0)
#   # print("new_matrix",new_matrix.shape)
#   # #print("c_hat",c_hat.shape)
#   w_hat = kron(new_matrix, c_hat)
  print("Reconstruction error: {}".format(
   round((torch.linalg.norm((w.reshape(-1) - w_hat.reshape(-1))).detach().numpy()).item(), 4)
  ))