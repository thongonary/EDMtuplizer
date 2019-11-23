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
from torch.utils.data.sampler import SubsetRandomSampler

import pickle
import argparse
import h5py
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
        self.pfjets_px = ((self.pfjets_px[:] - self.px_mean)/self.px_std).astype(np.float32)
        self.pfjets_py = ((self.pfjets_py[:] - self.py_mean)/self.py_std).astype(np.float32)
        self.pfjets_pz = ((self.pfjets_pz[:] - self.pz_mean)/self.pz_std).astype(np.float32)
        self.pfjets_energy = ((self.pfjets_energy[:] - self.energy_mean)/self.energy_std).astype(np.float32)
        
        self.calojets_px = ((self.calojets_px[:] - self.px_mean)/self.px_std).astype(np.float32)
        self.calojets_py = ((self.calojets_py[:] - self.py_mean)/self.py_std).astype(np.float32)
        self.calojets_pz = ((self.calojets_pz[:] - self.pz_mean)/self.pz_std).astype(np.float32)
        self.calojets_energy = ((self.calojets_energy[:] - self.energy_mean)/self.energy_std).astype(np.float32)


    def __len__(self):
        return len(self.pfjets_px)

    def __getitem__(self, idx):
        calojets = np.concatenate((self.calojets_px[idx].reshape(-1,1),
                              self.calojets_py[idx].reshape(-1,1),
                              self.calojets_pz[idx].reshape(-1,1),
                              self.calojets_energy[idx].reshape(-1,1)), axis=1)

        pfjets = np.concatenate((self.pfjets_px[idx].reshape(-1,1),
                              self.pfjets_py[idx].reshape(-1,1),
                              self.pfjets_pz[idx].reshape(-1,1),
                              self.pfjets_energy[idx].reshape(-1,1)), axis=1)

        return (calojets, pfjets)

class AutoEncoder(nn.Module):
    def __init__(self, features, latent_dim=2):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
                nn.Linear(features, int(features*2)),
                nn.ReLU(True),
                nn.Linear(int(features*2), int(features)),
                nn.ReLU(True),
                nn.Linear(int(features), latent_dim),
                nn.ReLU(True)
                )
        self.decoder = nn.Sequential(
                nn.Linear(latent_dim, int(features)),
                nn.ReLU(True),
                nn.Linear(int(features), int(features*2)),
                nn.ReLU(True),
                nn.Linear(int(features*2), features),
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

def train(model, train_loader, val_loader, epoch, loss_function, optimizer, scheduler): 
    model.train()
    loop = tqdm(train_loader)
    for data, target in loop:
        if torch.cuda.is_available():
            data, target = (data.cuda(), 
                            target.cuda()) 
        
        optimizer.zero_grad()
        prediction = model(data)
        
        loss = loss_function(prediction, target)
        losses['train'].append(loss.item())
        
        loss.backward()
        optimizer.step()
        
        loop.set_description('Epoch {}/{}'.format(epoch + 1, n_epochs))
        loop.set_postfix(loss=loss.item())
    del loop 
    model.eval()
    loop = tqdm(val_loader, ncols=200)
    for data, target in loop:
        if torch.cuda.is_available():
            data, target = (data.cuda(), 
                            target.cuda()) 
        
        prediction = model(data)
        
        loss = loss_function(prediction, target)
        losses['val'].append(loss.item())
        
        loop.set_description('Validation')
        loop.set_postfix(loss=loss.item())
    scheduler.step(loss)
    del loop

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Autoencoder for PF regression.')
    parser.add_argument("-i", '--input', default="step3_jets.h5", help='Input dataset')
    parser.add_argument("-b", '--batch-size', default=256, help='Batch size')
    parser.add_argument("-e", '--epochs', default=200, help='Number of epochs')
    parser.add_argument("-v", '--val-frac', default=0.2, help='Validation set fraction')
    parser.add_argument("-s", '--random-seed', default=12, help='Random seed')
    args = parser.parse_args()
    
    n_epochs = args.epochs

    data = JetDataset(args.input)
    num_train = len(data)
    indices = list(range(num_train))
    split = int(np.floor(args.val_frac * num_train))

    np.random.seed(args.random_seed)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
                        data, batch_size=args.batch_size, sampler=train_sampler,
                        )
    val_loader = torch.utils.data.DataLoader(
                        data, batch_size=args.batch_size, sampler=valid_sampler
                        )

    model = AutoEncoder(features=4)
    if torch.cuda.is_available():
        model = model.cuda()

    criterion = torch.nn.L1Loss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 
                                  mode='min',
                                  factor=0.3,
                                  patience=5,
                                  verbose=1,
                                  threshold=1e-4,
                                  cooldown=2,
                                  min_lr=1e-7
                                 )
    losses = {'train': [], 'val': []}
    for epoch in range(n_epochs):
        train(model=model, train_loader=train_loader,
                val_loader=val_loader, 
                loss_function=criterion,
                optimizer=optimizer,
                epoch=epoch,
                scheduler=scheduler)
    model.save("test.torch")
    # TODO: Tensorboard to keep the loss history
