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

net = BernoulliMLP(10,10,10)
criterion = nn.MSELoss()

net.zero_grad()

z = torch.randn(1,10)
y = net(z)

target = torch.ones_like(z)
loss = criterion(y, target)

print(net.fc1.bias.data)
loss.backward()

optimizer = torch.optim.SGD(net.parameters(), lr=1)
optimizer.step()
print(net.fc1.bias.data)