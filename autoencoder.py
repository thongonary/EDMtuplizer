import os
import sys
import torch
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

import pickle
import argparse

import time

class JetDataset(Dataset):
    def __init__(self, filename):
        self.infile = h5py.File(filename, "r")
        self.pfjets_px = self.infile["pfjets_px"]
        self.pfjets_py = self.infile["pfjets_py"]
        self.pfjets_pz = self.infile["pfjets_pz"]
        self.pfjets_energy = self.infile["pfjets_energy"]

        self.calojets_px = self.infile["calojets_px"]
        self.calojets_py = self.infile["calojets_py"]
        self.calojets_pz = self.infile["calojets_pz"]
        self.calojets_energy = self.infile["calojets_energy"]
        
        # Normalization 
        self.get_norm()
        self.set_norm()

    def get_norm(self):
        self.px_mean = np.mean(self.pfjets_px[:].flatten())
        self.px_std = np.std(self.pfjets_px[:].flatten())
        self.py_mean = np.mean(self.pfjets_py[:].flatten())
        self.py_std = np.std(self.pfjets_py[:].flatten())
        self.pz_mean = np.mean(self.pfjets_pz[:].flatten())
        self.pz_std = np.std(self.pfjets_pz[:].flatten())
        self.energy_mean = np.mean(self.pfjets_energy[:].flatten())
        self.energy_std = np.std(self.pfjets_energy[:].flatten())

    def set_norm(self):
        self.pfjets_px = (self.pfjets_px[:] - self.px_mean)/self.px_std
        self.pfjets_py = (self.pfjets_py[:] - self.py_mean)/self.py_std
        self.pfjets_pz = (self.pfjets_pz[:] - self.pz_mean)/self.pz_std
        self.pfjets_energy = (self.pfjets_energy[:] - self.energy_mean)/self.energy_std
        
        self.calojets_px = (self.calojets_px[:] - self.px_mean)/self.px_std
        self.calojets_py = (self.calojets_py[:] - self.py_mean)/self.py_std
        self.calojets_pz = (self.calojets_pz[:] - self.pz_mean)/self.pz_std
        self.calojets_energy = (self.calojets_energy[:] - self.energy_mean)/self.energy_std

    def __len__(self):
        return len(self.pfjets_pt)

    def __getitem__(self, idx):
        calojets = np.concatenate((self.calojets_px[idx],
                              self.calojets_py[idx],
                              self.calojets_pz[idx],
                              self.calojets_energy[idx]))

        pfjets = np.concatenate((self.pfjets_px[idx],
                              self.pfjets_py[idx],
                              self.pfjets_pz[idx],
                              self.pfjets_energy[idx]))

        return (calojets, pfjets)

class AutoEncoder(nn.Module):
    def __init__(self, features, latent_dim=2):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
                nn.Linear(features, int(features*3/2)),
                nn.ReLU(True),
                nn.Linear(int(features*3/2), int(features*2/3)),
                nn.ReLU(True)
                nn.Linear(int(features*2/3), latent_dim),
                nn.ReLU(True)
                )
        self.decoder = nn.Sequential(
                nn.Linear(latent_dim, int(features*2/3)),
                nn.ReLU(True),
                nn.Linear(int(features*2/3), int(features*3/2)),
                nn.ReLU(True)
                nn.Linear(int(features*3/2), features),
            )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def save(self, PATH):
        print("Saving the model\'s state_dict to {}".format(PATH))
        torch.save(self.state_dict(), PATH)

    def load(self, PATH):
        print("Loading the model\'s state_dict from {}".format(PATH))
        self.load_state_dict(torch.load(PATH))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Autoencoder for PF regression.')
    parser.add_argument("-i", '--input', default="step3.h5", help='Input dataset')
    parser.add_argument("-b", '--batch-size', default=256, help='Batch size')
    args = parser.parse_args()

    data = JetDataset(args.input)
    dataloader = DataLoader(data, batch_size=args.batch_size, 
                            num_worker=2, shuffle=True)



