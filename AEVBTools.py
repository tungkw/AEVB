import numpy as np
import torch
import torch.nn as nn

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

class GaussianMLP_forFrayFace(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(GaussianMLP_forFrayFace, self).__init__()
        self.gaussian_mlp = GaussianMLP(input_size, output_size, hidden_size)

    def forward(self, z):
        mu, log_sigma_sq = self.gaussian_mlp(z)
        mu = torch.sigmoid(mu)
        return mu, log_sigma_sq


if __name__ == "__main__":
    x = torch.from_numpy(np.ones((2,10))).float()
    model = GaussianMLP(10,5,10)
    y1,y2 = model(x)
    print(y1)
    print(y2)