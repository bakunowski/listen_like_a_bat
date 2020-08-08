import torch
from torch import nn
import torch.nn.functional as F


class TenInputsNet(nn.Module):

    def __init__(self, num_inputs):
        super(TenInputsNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(
            in_channels=16 * num_inputs, out_channels=32, kernel_size=3, stride=1)

        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1)

        self.conv4 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1)

        # self.linear1 = nn.Linear(in_features=38400, out_features=64)
        self.linear1 = nn.Linear(in_features=1920, out_features=64)
        self.linear2 = nn.Linear(in_features=64, out_features=32)
        self.linear3 = nn.Linear(in_features=32, out_features=12)

    def forward(self, a):
        x = []

        for i in range(len(a[1])):
            x.append(self.convolve(a[:, i]))

        x = torch.cat(x[:], 1)
        x = self.conv2(x)
        x = self.pool(F.relu(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def convolve(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        return x
