import torch
import numpy as np
import pickle
import sys
import os
import argparse
from pkl2pqvects import get_vects, get_phones
from model_spectral_attack import Spectral_attack

# Get command line arguments
commandLineParser = argparse.ArgumentParser()
commandLineParser.add_argument('PKL', type=str, help='Specify input pkl file')
commandLineParser.add_argument('ATTACK_MODEL', type=str, help='Specify trained attack model to load')
commandLineParser.add_argument('--N', default=225, type=int, help='Specify number of speakers')
commandLineParser.add_argument('--F', default=1000, type=int, help='Specify maximum number of frames in phone instance')

args = commandLineParser.parse_args()
pkl_file = args.PKL
attack_model_path = args.ATTACK_MODEL
N = args.N
F = args.F

# Save the command run
if not os.path.isdir('CMDs'):
    os.mkdir('CMDs')
with open('CMDs/eval_spectral_attack', 'a') as f:
    f.write(' '.join(sys.argv)+'\n')

pkl = pickle.load(open(pkl_file, "rb"))
print("Loaded pkl")

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

# Load the trained attack model
attack_model = torch.load(attack_model_path)
attack_model.eval()

# Get the predicted grades
y_pred = attack_model(X1, X2, M1, M2)
y_pred[y_pred<0.0]=0.0
y_pred[y_pred>6.0]=6.0
avg = torch.mean(y_pred)
print("Attacked average of ", avg)

# Future get the no attack average too 
