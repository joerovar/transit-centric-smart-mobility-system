from __future__ import division
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
from utils import generate_dataset, get_normalized_adj, get_Laplace, calculate_random_walk_matrix,nb_nll_loss
from model import *
import random,os,copy
import math
import tqdm
from scipy.stats import nbinom
import pickle as pk
import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
# Parameters
torch.manual_seed(0)
device = torch.device('cuda')
num_timesteps_output = 4
num_timesteps_input = num_timesteps_output

# Load dataset
A = np.load('../../data/02_INTERMEDIATE/demand_intermediate/A_o.npy')
X = np.load('../../data/02_INTERMEDIATE/demand_intermediate/training_data_15min.npy')

space_dim = X.shape[1]
batch_size = 4
hidden_dim_s = 32
hidden_dim_t = 7
rank_s = 20
rank_t = 4

epochs = 100

# Initial networks
TCN1 = B_TCN(space_dim, hidden_dim_t, kernel_size=3).to(device=device)
TCN2 = B_TCN(hidden_dim_t, rank_t, kernel_size = 3, activation = 'linear').to(device=device)
TCN3 = B_TCN(rank_t, hidden_dim_t, kernel_size= 3).to(device=device)
TNB = NBNorm(hidden_dim_t,space_dim).to(device=device)
SCN1 = D_GCN(num_timesteps_input, hidden_dim_s, 2).to(device=device)
SCN2 = D_GCN(hidden_dim_s, rank_s, 2, activation = 'linear').to(device=device)
SCN3 = D_GCN(rank_s, hidden_dim_s, 2).to(device=device)
SNB = NBNorm(hidden_dim_s,num_timesteps_output).to(device=device)
STmodel = ST_NB(SCN1, SCN2, SCN3, TCN1, TCN2, TCN3, SNB,TNB).to(device=device)

# Process data set
X = X.T
X = X.astype(np.float32)
X = X.reshape((X.shape[0],1,X.shape[1]))
split_line1 = int(X.shape[2] * 0.6)
split_line2 = int(X.shape[2] * 0.7)

print(X.shape,A.shape)
# normalization

train_original_data = X[:, :, :split_line1]
val_original_data = X[:, :, split_line1:split_line2]
test_original_data = X[:, :, split_line2:]
training_input, training_target = generate_dataset(train_original_data,
                                                    num_timesteps_input=num_timesteps_input,
                                                    num_timesteps_output=num_timesteps_output)
val_input, val_target = generate_dataset(val_original_data,
                                            num_timesteps_input=num_timesteps_input,
                                            num_timesteps_output=num_timesteps_output)
test_input, test_target = generate_dataset(test_original_data,
                                            num_timesteps_input=num_timesteps_input,
                                            num_timesteps_output=num_timesteps_output)
print('input shape: ',training_input.shape,val_input.shape,test_input.shape)

A_wave = get_normalized_adj(A)
A_q = torch.from_numpy((calculate_random_walk_matrix(A_wave).T).astype('float32'))
A_h = torch.from_numpy((calculate_random_walk_matrix(A_wave.T).T).astype('float32'))
A_q = A_q.to(device=device)
A_h = A_h.to(device=device)
# Define the training process

optimizer = optim.Adam(STmodel.parameters(), lr=1e-3)
training_nll   = []
validation_nll = []
validation_mae = []

for epoch in range(epochs):
    ## Step 1, training
    """
    # Begin training, similar training procedure from STGCN
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    """
    permutation = torch.randperm(training_input.shape[0])
    epoch_training_losses = []
    for i in range(0, training_input.shape[0], batch_size):
        # print(i)
        STmodel.train()
        optimizer.zero_grad()

        indices = permutation[i:i + batch_size]
        X_batch, y_batch = training_input[indices], training_target[indices]
        X_batch = X_batch.to(device=device)
        

        n_train,p_train = STmodel(X_batch,A_q,A_h)
        y_batch = y_batch.to(device=device)
        loss = nb_nll_loss(y_batch,n_train,p_train)
        loss.backward()
        optimizer.step()
        epoch_training_losses.append(loss.detach().cpu().numpy())
    training_nll.append(sum(epoch_training_losses)/len(epoch_training_losses))
    ## Step 2, validation
    print('Epoch: {}'.format(epoch))
    print("Training loss: {}".format(training_nll[-1]))
    if np.asscalar(training_nll[-1]) == min(training_nll):
        best_model = copy.deepcopy(STmodel.state_dict())
    checkpoint_path = "checkpoints/"
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    with open("checkpoints/losses.pk", "wb") as fd:
        pk.dump((training_nll), fd)
    if np.isnan(training_nll[-1]):
        break
STmodel.load_state_dict(best_model)
torch.save(STmodel,'../../data/04_MODEL/STNB_route81_15min.pth')
