'''

Takes the information in the pkl file and prepares matrices X1, X2, y
X = [N x P*(P-1)*0.5 x I x F x n]
y = [N] -> the reference grade for the speaker

N = number of speakers (e.g. 993)
P = number of phones (e.g. 47)
I = maximum number of phone instances (e.g. 500)
F = maximum number of frames per phone instance (e.g. 100)
n = number of features (e.g. 13)

IN THIS FILE THE I-dimension IS REMOVED
'''
import numpy as np
import pickle
import argparse
import sys
import os

def get_phones(alphabet='arpabet'):
    if alphabet == 'arpabet':
        vowels = ['aa', 'ae', 'eh', 'ah', 'ea', 'ao', 'ia', 'ey', 'aw', 'ay', 'ax', 'er', 'ih', 'iy', 'uh', 'oh', 'oy', 'ow', 'ua', 'uw']
        consonants = ['el', 'ch', 'en', 'ng', 'sh', 'th', 'zh', 'w', 'dh', 'hh', 'jh', 'em', 'b', 'd', 'g', 'f', 'h', 'k', 'm', 'l', 'n', 'p', 's', 'r', 't', 'v', 'y', 'z'] + ['sil']
        phones = vowels + consonants
        return phones
    if alphabet == 'graphemic':
        vowels = ['a', 'e', 'i', 'o', 'u']
        consonants = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'y', 'z'] + ['sil']
        phones = vowels + consonants
        return phones
    raise ValueError('Alphabet name not recognised: ' + alphabet)


def get_vects(obj, phones, N, F=1000):
    n = len(obj['plp'][0][0][0][0][0]) # dimension of mfcc vector
    #N = len(obj['plp'])
    P = len(phones)-1

    # Define the tensors required
    X1 = np.zeros((N, int(P*(P-1)*0.5), F, n))
    X2 = np.zeros((N, int(P*(P-1)*0.5), F, n))
    # Define the masks required
    M1 = np.zeros((N, int(P*(P-1)*0.5), F, n))
    M2 = np.zeros((N, int(P*(P-1)*0.5), F, n))

    for spk in range(N):
        print("On speaker " + str(spk) + " of " + str(N))
        Xs = np.zeros((P, F, n))
        F_counter = np.zeros(P)

        for utt in range(len(obj['plp'][spk])):
            for w in range(len(obj['plp'][spk][utt])):
                for ph in range(len(obj['plp'][spk][utt][w])):
                    # n.b. this is iterating through the phones instances that occur sequentially in a word
                    ph_ind = obj['phone'][spk][utt][w][ph]
                    if F_counter[ph_ind] >= F:
                        continue
                    for frame in range(len(obj['plp'][spk][utt][w][ph])):
                        F_ind = int(F_counter[ph_ind].item())
                        if F_counter[ph_ind] >= F:
                            continue
                        X = np.array(obj['plp'][spk][utt][w][ph][frame])
                        Xs[ph_ind][F_ind] = X
                        F_counter[ph_ind] += 1


        # Consturct every unique pairing of phones, related mfcc vectors
        k = 0
        for i in range(P):
            for j in range(i + 1, P):
                # Make X1 and M1
                X1[spk][k] = Xs[i]
                F_ind = int(F_counter[i].item())
                M1[spk][k][:F_ind] = np.tile(np.ones(n), (F_ind, 1))

                # Make X2 and M2
                X1[spk][k] = Xs[j]
                F_ind = int(F_counter[j].item())
                M1[spk][k][:F_ind] = np.tile(np.ones(n), (F_ind, 1))

                k += 1

    return X1, X2, M1, M2

'''
# Get command line arguments
commandLineParser = argparse.ArgumentParser()
commandLineParser.add_argument('PKL', type=str, help='Specify pkl file')
commandLineParser.add_argument('OUT', type=str, help='Specify output pkl file')
commandLineParser.add_argument('--F', default=100, type=int, help='Specify maximum number of frames in phone instance')
commandLineParser.add_argument('--I', default=500, type=int, help='Specify maximum number of instances of a phone')


args = commandLineParser.parse_args()
pkl_file = args.PKL
out_file = args.OUT
F = args.F
I = args.I

# Save the command run
if not os.path.isdir('CMDs'):
    os.mkdir('CMDs')
with open('CMDs/pkl2pqvects.cmd', 'a') as f:
    f.write(' '.join(sys.argv)+'\n')

pkl = pickle.load(open(pkl_file, "rb"))
print("Loaded pkl")

# Get the phones
phones = get_phones()

# Get the batched tensors
X1, X2, M1, M2 = get_vects(pkl, phones, F, I)

# Get the output labels
y = (pkl['score'])

# Save to pickle file
pkl_obj = [X1.tolist(), X2.tolist(), M1.tolist(), M2.tolist(), y]
pickle.dump(pkl_obj, open(out_file, "wb"))
'''
