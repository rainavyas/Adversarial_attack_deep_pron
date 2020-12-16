import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import pickle
import sys
import os
import argparse
from pkl2pqvects import get_vects, get_phones
from model_spectral_attack import Spectral_attack
import math

def clip_params(model, barrier_val):
    old_params = {}

    for name, params in model.named_parameters():
        old_params[name] = params.clone()

    old_params['noise_root'][old_params['noise_root']>math.log(barrier_val)] = math.log(barrier_val)
    ''' 
    for i, param in enumerate(old_params['noise_root']):
        
        if param > math.log(barrier_val):
            old_params['noise_root'][i] = math.log(barrier_val)
    '''
        
    for name, params in model.named_parameters():
        params.data.copy_(old_params[name])


# Get command line arguments
commandLineParser = argparse.ArgumentParser()
commandLineParser.add_argument('PKL', type=str, help='Specify input pkl file')
commandLineParser.add_argument('OUT', type=str, help='Specify output pt file')
commandLineParser.add_argument('MODEL_PATH', type=str, help='Specify trained model to attack path')
commandLineParser.add_argument('--e', default=1.0, type=float, help='Specify constraint on size of attack')
commandLineParser.add_argument('--N', default=993, type=int, help='Specify number of speakers')
commandLineParser.add_argument('--F', default=100, type=int, help='Specify maximum number of frames in phone instance')
commandLineParser.add_argument('--checkpoint', default=None, type=str, help='Specify part trained model path')

args = commandLineParser.parse_args()
pkl_file = args.PKL
out_file = args.OUT
model_path = args.MODEL_PATH
e = args.e
N = args.N
F = args.F
checkpoint = args.checkpoint

# Save the command run
if not os.path.isdir('CMDs'):
    os.mkdir('CMDs')
with open('CMDs/training_spectral_attack', 'a') as f:
    f.write(' '.join(sys.argv)+'\n')

pkl = pickle.load(open(pkl_file, "rb"))
print("Loaded pkl")

# Get the phones
phones = get_phones()

# Get the batched tensors
X1, X2, M1, M2 = get_vects(pkl, phones, N, F)

# Get the output labels
y = (pkl['score'])

# Convert to tensors
X1 = torch.from_numpy(X1).float()
X2 = torch.from_numpy(X2).float()
M1 = torch.from_numpy(M1).float()
M2 = torch.from_numpy(M2).float()
y = torch.FloatTensor(y)


# Split into training and validation sets
validation_size = 50
X1_train = X1[validation_size:]
X1_val = X1[:validation_size]
X2_train = X2[validation_size:]
X2_val = X2[:validation_size]
M1_train = M1[validation_size:]
M1_val = M1[:validation_size]
M2_train = M2[validation_size:]
M2_val = M2[:validation_size]
y_train = y[validation_size:N]
y_val = y[:validation_size]

# Define training constants
lr = 8*1e-3
epochs = 20
bs = 50
sch = 0.985
seed = 1
torch.manual_seed(seed)
spectral_dim = 24
mfcc_dim = 13

#init_root = torch.FloatTensor([5]*spectral_dim)
init_root = torch.randn(X1_train.size(2), spectral_dim)

# Store all training dataset in a single wrapped tensor
train_ds = TensorDataset(X1_train, X2_train, M1_train, M2_train)

# Use DataLoader to handle minibatches easily
train_dl = DataLoader(train_ds, batch_size = bs, shuffle = True)

# Initialise deep pron model to be trained
if checkpoint == None:
    attack_model = Spectral_attack(spectral_dim, mfcc_dim, model_path, init_root)
else:
    attack_model = torch.load(checkpoint)

print("Initialised model")

optimizer = torch.optim.SGD(attack_model.parameters(), lr=lr, momentum = 0.9, nesterov=True)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = sch)

for epoch in range(epochs):
    attack_model.train()
    print("On Epoch, ", epoch)

    for x1, x2, m1, m2 in train_dl:

        # Forward pass
        y_pred = attack_model(x1, x2, m1, m2)

        # Compute loss
        loss = -1*torch.sum(y_pred)

        # Zero gradients, backward pass, update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("running avg: ", (-1*loss/bs))

        # Keep weights below barrier
        clip_params(attack_model, e)

    # Validation
    attack_model.eval()
    y_val_pred = attack_model(X1_val, X2_val, M1_val, M2_val)
    y_val_pred[y_val_pred>6.0]=6.0
    y_val_pred[y_val_pred<0.0]=0.0
    avg = torch.sum(y_val_pred)/validation_size
    print("Validation Avg: ", avg)
'''
    old_params = {}
    for name, params in attack_model.named_parameters():
        old_params[name] = params.clone()

    for i, param in enumerate(old_params['noise_root']):
        c = math.exp(param)
        print("Channel:", i, " ", c)
'''
# Save the trained model
torch.save(attack_model, out_file)
