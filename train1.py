import time
import torch
from torch import nn
from utils import to_variable
from torch.nn import DataParallel
from model import TenInputsNet
from dataset import EchoesDataset, show_echo_batch
from torch.utils.data import DataLoader, random_split
# from torch.utils.tensorboard import SummaryWriter

# Csv file with folder/name of audio file and class
labelscv = "./listen_like_a_bat.csv"

# Path to folder with folders containing .csv files with impulse responses
folder = "./ShuffledData"

# Path to bat call
call_datapath = "./BatCalls/Glosso_call.wav"

# Hyperparameters
num_epochs = 1000
learning_rate = 0.003
train_dataset = 0
test_dataset = 0
is_cuda = False

# writer = SummaryWriter('./logdir')

model = TenInputsNet()

trainset, valset = random_split(EchoesDataset(labelscv, call_datapath),
                                [2000, 349])
print(len(trainset))
print(len(valset))

train_loader = DataLoader(trainset, batch_size=16, shuffle=True)
val_loader = DataLoader(valset, batch_size=16, shuffle=True)

if torch.cuda.is_available():
    model = torch.nn.DataParallel(model)
    is_cuda = True

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(train_loader)
print("Number of batches: ", total_step)
start_time = time.time()


for epoch in range(num_epochs):

    print('Epoch {}/{}'.format(epoch+1, num_epochs))
    print('-' * 10)

    for i, sample_batched in enumerate(train_loader):
        running_loss = 0.0
        running_corrects = 0

        if type(model) is DataParallel:
            model = model.cuda()
        else:
            model = model

        labels, echoes = list(
            map(lambda x: to_variable(x, is_cuda=is_cuda), sample_batched))

        # print(labels.size(), echoes.size())
        # print("Input 1:", echoes[:, 0].size())
        # 10, 1, w, h
        # print(echoes.size())
        # show_echo_batch(labels, echoes)
        # show_echo_batch(labels[1], echoes[0][:, 1])
        # show_echo_batch(labels[2], echoes[0][:, 2])

        echoes = echoes.float()
        optimizer.zero_grad()
        # print("0: ", echoes[:, 0])
        # print("1: ", echoes[:, 1])

        outputs = model(echoes[:, 0], echoes[:, 1], echoes[:, 2],
                        echoes[:, 3], echoes[:, 4], echoes[:, 5],
                        echoes[:, 6], echoes[:, 7], echoes[:, 8], echoes[:, 9])
        _, preds = torch.max(outputs.data, 1)

        # print("Outputs: ", outputs)
        print("Labels: ", labels)
        print("Preds: ", preds)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)
        if i % 10 == 0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f acc: %.1f' %
                  (epoch + 1, i + 1, running_loss / 10, running_corrects / 10))
            running_loss = 0.0
            running_corrects = 0

    with torch.no_grad():

        for i, ds in enumerate(val_loader):
            running_loss = 0.0
            running_corrects = 0

            if type(model) is DataParallel:
                model = model.cuda()
            else:
                model = model

            labels, echoes = list(
                map(lambda x: to_variable(x, is_cuda=is_cuda), ds))

            echoes = echoes.float()
            outputs = model(echoes[:, 0], echoes[:, 1], echoes[:, 2], echoes[:, 3], echoes[:, 4],
                            echoes[:, 5], echoes[:, 6], echoes[:, 7], echoes[:, 8], echoes[:, 9])
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data)

            if i % 10 == 0:    # print every 2000 mini-batches
                print('Test: [%d, %5d] loss: %.3f acc: %1d' %
                      (epoch + 1, i + 1, running_loss / 10, running_corrects / 10))

