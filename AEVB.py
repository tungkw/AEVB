import numpy as np
import torch
import torch.nn as nn
import setting
import dataset
from torch.utils.data import DataLoader
from torch.distributions import Normal
from AEVBTools import *

class AEVB(nn.Module):
    def __init__(self, data_size, input_size, latent_size, data='FreyFace'):
        super(AEVB, self).__init__()
        self.data_size = data_size
        self.input_size = input_size
        self.latent_size = latent_size
        self.encoder = GaussianMLP(input_size, latent_size ,setting.hidden_size)
        self.decoder = GaussianMLP_forFrayFace(latent_size, input_size ,setting.hidden_size)
        self.normal = Normal(torch.zeros(self.latent_size), torch.ones(self.latent_size))
    
    def forward(self, x):
        mu_z, log_sigma_z = self.encoder(x)
        sigma_z = torch.exp(log_sigma_z)

        lower_bound = torch.tensor([0]).float()

        e = self.normal.sample()
        z = self.gTheta(mu_z, sigma_z, e)
        
        mu_x, log_sigma_x = self.decoder(z)
        sigma_x = torch.exp(log_sigma_x)
        log_prob = torch.distributions.Normal(mu_x, sigma_x).log_prob(x)

        kl_div = 0.5 * torch.sum(1 + torch.log(torch.pow(sigma_z,2)) - torch.pow(mu_z,2) - torch.pow(sigma_z,2), dim=-1)
        reconstruct_error = torch.norm(log_prob, dim=-1)

        lower_bound = torch.sum(kl_div + reconstruct_error) * (self.data_size/x.size()[0])
        return lower_bound

    def gTheta(self, mu, sigma, e):
        return mu + sigma * e

    def GaussianPDF(self, mu, sigma, x):
        sigma_mat = torch.eye(self.input_size) * sigma
        print(sigma_mat.size())
        sigma_inv = torch.inverse(sigma_mat)
        sigma_det = torch.det(sigma_mat)
        coef = 1/(torch.pow(2*np.pi, self.input_size/2) * torch.pow(sigma_det, 0.5))
        return  coef * torch.exp(-0.5 * torch.matmul(torch.matmul(torch.transpose(x-mu), sigma_inv),x-mu))

if __name__ == "__main__":
    dataset = dataset.FreyFaceDataset(setting.freyface_path)
    data_loader = DataLoader(dataset, batch_size=setting.M, shuffle=True, num_workers=4)

    model = AEVB(dataset.data_size, dataset.sample_size, setting.latent_size, data='FreyFace')
    print(model)
    optimizer = torch.optim.Adagrad(model.parameters())

    for _ in range(1):
        for batch in data_loader:
            model.zero_grad()
            lower_bound = model(batch)
            optimizer.step()
            print(lower_bound.requires_grad)
            print(lower_bound.grad_fn)
            print(lower_bound.grad_fn.next_functions[0][0])
            print(lower_bound.grad_fn.next_functions[0][0].next_functions[0][0])
            print(lower_bound.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0])
            print(lower_bound.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions[0][0])
            # for parameter in model.parameters():
            #     print(parameter.grad)
            break





# decoder = GaussianMLP(, hidden_size, )

# net = BernoulliMLP(10,10,10)
# criterion = nn.MSELoss()

# net.zero_grad()

# z = torch.randn(1,10)
# y = net(z)

# target = torch.ones_like(z)
# loss = criterion(y, target)

# print(net.fc1.bias.data)
# loss.backward()

# optimizer = torch.optim.SGD(net.parameters(), lr=1)
# optimizer = torch.optim.Adagrad(net.parameters(), lr=1)
# optimizer.step()
# print(net.fc1.bias.data)
