import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable



class BernoulliMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(BernoulliMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, z):
        z = self.fc1(z)
        z = torch.tanh(z)
        z = self.fc2(z)
        z = torch.sigmoid(z)
        return z



class GaussianMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(GaussianMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, z):
        z = self.fc1(z)
        z = torch.tanh(z)
        mu = self.fc2(z)
        log_sigma_sq = self.fc3(z)
        return mu, log_sigma_sq



class AEVB(nn.Module):
    def __init__(self, sample_dim, latent_dim, hidden_dim, data='FreyFace'):
        super(AEVB, self).__init__()
        self.data = data
        self.sample_dim = sample_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.encoder = GaussianMLP(self.sample_dim, self.latent_dim ,self.hidden_dim)
        if self.data == 'FreyFace':
            self.decoder = GaussianMLP(self.latent_dim, self.sample_dim ,self.hidden_dim)
        elif self.data == 'MINST':
            self.decoder = BernoulliMLP(self.latent_dim, self.sample_dim ,self.hidden_dim)
    

    def forward(self, x):
        '''q(z|x)'''
        mu_z, log_var_sqr_z = self.encoder(x)
        z = self.sample_z(mu_z, log_var_sqr_z)


        '''p(x|z)'''
        if self.data == 'FreyFace':
            mu_x, log_var_sqr_x = self.decoder(z)
            mu_x = torch.sigmoid(mu_x)
            # log(p(x)) = log(1) - log((2*pi)^0.5) - log(var) + (-0.5*(x-mu)^2/var^2)
            log_prob_x = -0.5*torch.tensor(2*np.pi).log() - 0.5*log_var_sqr_x - 0.5*(x-mu_x)**2/log_var_sqr_x.exp()
        elif self.data == 'MINST':
            y = self.decoder(z)
            log_prob_x = x*y.log() + (1-x)*(1-y).log()
        log_prob_x = log_prob_x.sum(dim=-1)


        '''KL divergence'''
        # 0.5 * SUM(1 + log(var^2) - mu^2 - var^2)
        kl_div = 0.5 * torch.sum(1 + log_var_sqr_z - mu_z**2 - log_var_sqr_z.exp(), dim=-1)


        '''reconstruct error'''
        # L = 1 for (1/L)*SUM(log(p(x|z)))
        re_err = log_prob_x


        '''variational lower bound'''
        lower_bound = (kl_div + re_err).mean()
        return lower_bound



    def sample_z(self, mu, log_var_sqr):
        e = Variable(torch.randn(mu.size()[0], self.latent_dim))
        e = e.to(device=torch.cuda.current_device())
        return self.gTheta(mu, log_var_sqr, e)


    '''g(x,e)'''
    def gTheta(self, mu, log_var_sqr, e):
        # z = mu + var * e
        return mu + torch.exp(0.5 * log_var_sqr) * e
