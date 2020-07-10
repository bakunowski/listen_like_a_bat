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

    def forward(self, a):
        # print("First convolution", a.size())
        x = []

        for i in range(len(a[1])):
            x.append(self.convolve(a[:, i]))

        # print("In network after maxpool1: ", len(x))
        # 4, 16, 63, 82
        x = torch.cat(x[:], 1)
        # print("In network after cat: ", x.size())
        # 4, 160, 63, 82

        # print("In network after cat: ", x.size())
        x = self.conv2(x)
        # print("In network second conv: ", x.size())
        x = self.maxpool2(x)
        # print("In network pooled: ", x.size())
        x = torch.flatten(x, start_dim=1)
        # print("In network flat: ", x.size())
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        # print("In network a: ", x.size())
        return x

#     def forward(self, input1, input2):
#         c = self.conv(input1)
#         f = self.fc1(input2)
#         # now we can reshape `c` and `f` to 2D and concat them
#         combined = torch.cat((c.view(c.size(0), -1),
#                               f.view(f.size(0), -1)), dim=1)
#         out = self.fc2(combined)
#         return out

    def convolve(self, x):
        print("In network before conv1: ", x.size())
        # 4, 1, 129, 167
        x = self.conv1(x)
        print("In network after conv1: ", x.size())
        # 4, 16, 127, 165
        x = self.maxpool1(x)
        return x
