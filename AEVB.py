import numpy as np
import torch
import torch.nn as nn
import setting
import dataset
from torch.utils.data import DataLoader
from torch.distributions import Normal
from AEVBTools import *

import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class AEVB(nn.Module):
    def __init__(self, data_size, input_size, latent_size, data='FreyFace'):
        super(AEVB, self).__init__()
        self.data = data
        self.data_size = data_size
        self.input_size = input_size
        self.latent_size = latent_size
        self.encoder = GaussianMLP(input_size, latent_size ,setting.hidden_size)
        self.decoder = GaussianMLP(latent_size, input_size ,setting.hidden_size)
    
    def forward(self, x):
        mu_z, log_sigma_sqr_z = self.encoder(x)
        sigma_z = torch.exp(0.5*log_sigma_sqr_z)

        e = torch.normal(mean=torch.zeros(self.latent_size),std=torch.ones(self.latent_size)).to(device=device)
        z = self.gTheta(mu_z, sigma_z, e)
        
        mu_x, log_sigma_sqr_x = self.decoder(z)
        sigma_x = torch.exp(0.5*log_sigma_sqr_x)
        # if self.data == 'FreyFace':
        #     mu_x = torch.sigmoid(mu_x)
        #     sigma_x = torch.sigmoid(sigma_x)
        # log_prob_x = torch.distributions.Normal(loc=mu_x, scale=sigma_x**2).log_prob(x).sum(dim=-1)
        log_prob_x = (- 0.5 * torch.log(torch.tensor(2*np.pi)) - 0.5 * (x - mu_x)**2 / sigma_x**2 ).sum(dim=-1)
        print(mu_x.mean())
        print(sigma_x.mean())
        print(x.mean())
        # print(log_prob_x)
        print(torch.exp(log_prob_x).mean())

        kl_div = 0.5 * torch.sum(1 + log_sigma_sqr_z - mu_z**2 - sigma_z**2, dim=-1)
        reconstruct_error = log_prob_x
        # reconstruct_error = torch.nn.CrossEntropyLoss(x,mu_x)

        lower_bound = (kl_div + reconstruct_error).mean() * self.data_size
        # print(kl_div.mean())
        print(reconstruct_error.mean())
        # print(lower_bound.mean())
        return -lower_bound

    def gTheta(self, mu, sigma, e):
        return mu + sigma * e

if __name__ == "__main__":
    dataset = dataset.FreyFaceDataset(setting.freyface_path)
    data_loader = DataLoader(dataset, batch_size=setting.M, shuffle=True, num_workers=4)

    model = AEVB(dataset.data_size, dataset.sample_size, setting.latent_size, data='FreyFace')
    model.to(device=device)

    print(model)

    # initialize
    for para in model.parameters():
        torch.nn.init.normal_(para,0,0.01)

    # train
    optimizer = torch.optim.Adagrad(model.parameters(),lr=0.01)
    record = []
    for k in range(10):
        tmp = 0
        for i,batch in enumerate(data_loader):
            model.zero_grad()
            lower_bound = model(batch.to(device=device))
            lower_bound.backward()
            optimizer.step()
            # print(lower_bound)
            tmp += lower_bound.data
            # record.append([i,lower_bound.data])
            # break
        print("epoch",k,tmp)
        tmp /= dataset.data_size
        record.append(tmp)
    # plt.plot(record)
    # plt.show()