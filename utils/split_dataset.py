import torch
from math import *

path="data/"

with open(path+'data.csv','r') as fichier:
    lines=fichier.readlines()

entete=lines[0]
del(lines[0])
nb_lines=len(lines)
size_test=floor(0.2*nb_lines)
size_train=ceil(0.8*nb_lines)
rp = torch.randperm(nb_lines).tolist()
test_idx=rp[:size_test]
train_idx=rp[-size_train:]


with open(path+'data_train.csv','w') as fichier:
    fichier.write(entete)
    for i in train_idx:
        content=lines[i]
        fichier.write(content)

with open(path+'data_test.csv','w') as fichier:
    fichier.write(entete)
    for i in test_idx:
        content=lines[i]
        fichier.write(content)