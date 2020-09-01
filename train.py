import torch
import numpy as np
from torch import nn
from utils import to_variable, show_echo_batch
from torch.nn import DataParallel
from model import TenInputsNet
from dataset import EchoesDataset
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

# Csv file with folder/name of audio file and class
labelscvTrain = "./listen_like_a_bat10Train_Interval10_valset.csv"
labelscvVal = "./listen_like_a_bat10Validation_Interval10_valset.csv"

# Path to folder with folders containing .csv files with impulse responses
folderTrain = "./Data/Train_Test_Validation/Train/Interval10"
folderVal = "./Data/Train_Test_Validation/Validation/Interval10"

# Path to bat call
call_datapath = "./BatCalls/Glosso_call.wav"

# Hyperparameters
num_epochs = 30
learning_rate = 0.0001
batch_size = 16

is_cuda = False
LOAD_CHECKPOINT = False

writer = SummaryWriter('./logdir_train')
writer2 = SummaryWriter('./logdir_val')

state_dict = {}
model = TenInputsNet(10)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

trainset = EchoesDataset(labelscvTrain, call_datapath, folderTrain)
valset = EchoesDataset(labelscvVal, call_datapath, folderVal)

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

if LOAD_CHECKPOINT:
    checkpoint = torch.load('/homes/kb316/FinalProject/Epoch_40_Loss_0.088536_Acc_97.531.pth')
    epoch = epoch + 40
    model.load_state_dict(checkpoint)

print('-' * 10, "DATA INFO", '-' * 10)
print("Number of batches in train: ", len(train_loader))
print("Number of batches in test: ", len(val_loader))
print('\n')

for epoch in range(num_epochs):   # loop over the dataset multiple times
    print('Epoch {}/{}'.format(epoch+1, num_epochs))
    print('-' * 10)

    running_loss = 0.0
    running_loss2 = 0.0
    total = 0
    correct = 0
    accuracy = 0
    accuracy2 = 0

    for i, sample_batched in enumerate(train_loader):

        if type(model) is DataParallel:
            model = model.cuda()
        else:
            model = model

        # get the inputs;
        labels, echoes = list(
            map(lambda x: to_variable(x, is_cuda=is_cuda), sample_batched))

        # show_echo_batch(labels, echoes)
        echoes = echoes.float()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(echoes)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # max value from each output in batch is the predicted class
        _, preds = torch.max(outputs.data, 1)

        # accuracy
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        accuracy += 100 * correct / total
        accuracy2 += 100 * correct / total

        writer.add_scalar('Loss/Train', loss.item(), (epoch*len(train_loader)) + i)
        writer.flush()

        # print statistics
        running_loss += loss.item()
        running_loss2 += loss.item()
        if i % 10 == 9:    # print every 10 mini-batches
            print('[%d, %5d] loss: %.3f acc: %.1f' %
                  (epoch + 1, i + 1, running_loss / 10, accuracy / 10))
            running_loss = 0.0
            total = 0
            correct = 0
            accuracy = 0

    print('MEAN: loss: %.3f acc: %.1f' % (running_loss2 / len(train_loader), accuracy2 / len(train_loader)))
    writer.add_scalar('PerEpoch/Loss', running_loss2 / len(train_loader), epoch)
    writer.add_scalar('PerEpoch/Accuracy', accuracy2 / len(train_loader), epoch)
    writer.flush()

    with torch.no_grad():
        running_loss = 0.0
        total = 0
        correct = 0
        accuracy = 0

        model.eval()
        for i, ds in enumerate(val_loader):

            if type(model) is DataParallel:
                model = model.cuda()
            else:
                model = model

            labels, echoes = list(
                map(lambda x: to_variable(x, is_cuda=is_cuda), ds))

            echoes = echoes.float()
            outputs = model(echoes)
            _, preds = torch.max(outputs.data, 1)

            # accuracy
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            accuracy += 100 * correct / total

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            writer2.add_scalar('Loss/Test', loss.item(), (epoch*len(val_loader)) + i)
            writer2.flush()

        print('Test: [%d, %5d] loss: %.3f acc: %1d' %
              (epoch + 1, i + 1, running_loss / len(val_loader), accuracy / len(val_loader)))
        writer2.add_scalar('PerEpoch/Loss', running_loss / len(val_loader), epoch)
        writer2.add_scalar('PerEpoch/Accuracy', accuracy / len(val_loader), epoch)
        writer2.flush()

    save_path = 'Epoch_%d_Loss_%.6f_Acc_%.3f.pth' % (
        epoch + 1, running_loss / len(val_loader), accuracy / len(val_loader))
    torch.save(model.state_dict(), save_path)
