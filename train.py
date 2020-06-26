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
batch_size = 16
is_cuda = False

# writer = SummaryWriter('./logdir')

state_dict = {}
model = TenInputsNet()

trainset, valset = random_split(EchoesDataset(labelscv, call_datapath),
                                [2000, 349])

train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                          num_workers=4)
val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False,
                        num_workers=4)

if torch.cuda.is_available():
    print('-' * 10, "GPU INFO", '-' * 10)
    print("Using GPUs")
    print("Number of GPUs: ", torch.cuda.device_count())
    model = torch.nn.DataParallel(model)
    is_cuda = True

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(train_loader)
print('-' * 10, "DATA INFO", '-' * 10)
print("Number of batches in train: ", total_step)
print("Number of batches in test: ", len(val_loader))
print('\n')
start_time = time.time()


for epoch in range(num_epochs):   # loop over the dataset multiple times

    print('Epoch {}/{}'.format(epoch+1, num_epochs))
    print('-' * 10)
    running_loss = 0.0
    total = 0
    correct = 0
    accuracy = 0

    for i, sample_batched in enumerate(train_loader):

        if type(model) is DataParallel:
            model = model.cuda()
        else:
            model = model

        # get the inputs; data is ...
        labels, echoes = list(
            map(lambda x: to_variable(x, is_cuda=is_cuda), sample_batched))

        # show_echo_batch(labels[1], echoes[0][:, 1])
        echoes = echoes.float()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(echoes[:, 0], echoes[:, 1], echoes[:, 2],
                        echoes[:, 3], echoes[:, 4], echoes[:, 5],
                        echoes[:, 6], echoes[:, 7], echoes[:, 8], echoes[:, 9])
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # max value from each output in batch is the predicted class
        _, preds = torch.max(outputs.data, 1)

        # accuracy
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        accuracy += 100 * correct / total

        #print("Outputs: ", outputs)
        #print("Labels: ", labels)
        #print("Preds: ", preds)
        #print("Num correct: ", torch.sum(preds == labels.data))
        #print("Percent correct: ", torch.sum(preds == labels.data).item() / batch_size)

        # print statistics
        running_loss += loss.item()
        if i % 25 == 24:    # print every 10 mini-batches
            print('[%d, %5d] loss: %.3f acc: %.1f' %
                  (epoch + 1, i + 1, running_loss / 25, accuracy / 25))
            running_loss = 0.0
            total = 0
            correct = 0
            accuracy = 0

    with torch.no_grad():
        running_loss = 0.0
        total = 0
        correct = 0
        accuracy = 0
        for i, ds in enumerate(val_loader):

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

            # accuracy
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            accuracy += 100 * correct / total

            #print("Labels: ", labels)
            #print("Preds: ", preds)

            loss = criterion(outputs, labels)
            running_loss += loss.item()
        print('Test: [%d, %5d] loss: %.3f acc: %1d' %
              (epoch + 1, i + 1, running_loss / len(val_loader), accuracy / len(val_loader)))
