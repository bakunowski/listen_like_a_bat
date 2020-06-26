import torch
from torch import nn


class TenInputsNet(nn.Module):
    def __init__(self):
        super(TenInputsNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=3, stride=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(
            in_channels=160, out_channels=32, kernel_size=3, stride=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.linear1 = nn.Linear(in_features=38400, out_features=64)
        self.linear2 = nn.Linear(in_features=64, out_features=32)
        self.linear3 = nn.Linear(in_features=32, out_features=12)

    def forward(self, a, b, c, d, e, f, g, h, i, j):
        #print("In network a: ", a.size())
        a = self.convolve(a)
        b = self.convolve(b)
        c = self.convolve(c)
        d = self.convolve(d)
        e = self.convolve(e)
        f = self.convolve(f)
        g = self.convolve(g)
        h = self.convolve(h)
        i = self.convolve(i)
        j = self.convolve(j)
        x = torch.cat((a, b, c, d, e, f, g, h, i, j), 1)
        x = self.conv2(x)
        #print("In network x: ", x.size())
        x = self.maxpool2(x)
        #print("In network pooled: ", x.size())
        x = torch.flatten(x, start_dim=1)
        #print("In network flat: ", x.size())
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        # print("In network a: ", x.size())
        return x

    def convolve(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        return x
