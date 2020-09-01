import torch
import numpy as np
from torch import nn
from utils import to_variable, show_echo_batch, plot_confusion_matrix
from torch.nn import DataParallel
from model import TenInputsNet
from dataset import EchoesDataset
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Csv file with folder/name of audio file and class
labelscvTest = "./datasetCSV/listen_like_a_bat10Test_Interval10_valset.csv"

# Path to folder with folders containing .csv files with impulse responses
folderTest = "./Data/Train_Test_Validation/Test/Interval10"

# Path to bat call
call_datapath = "./BatCalls/Glosso_call.wav"

# Hyperparameters
num_epochs = 1

is_cuda = False
LOAD_CHECKPOINT = True

model = TenInputsNet(10)

testset = EchoesDataset(labelscvTest, call_datapath, folderTest)

test_loader = DataLoader(testset, batch_size=len(testset), shuffle=True,
                        num_workers=4)

if torch.cuda.is_available():
    print('-' * 10, "GPU INFO", '-' * 10)
    print("Using GPUs")
    print("Number of GPUs: ", torch.cuda.device_count())
    model = torch.nn.DataParallel(model)
    is_cuda = True

if LOAD_CHECKPOINT:
    checkpoint = torch.load('/homes/kb316/listen_like_a_bat/FinalResults/ValSplit/Epoch_40_Loss_0.088536_Acc_97.531.pth')
    model.load_state_dict(checkpoint)


with torch.no_grad():
    total = 0
    correct = 0
    accuracy = 0

    model.eval()
    for i, ds in enumerate(test_loader):

        if type(model) is DataParallel:
            model = model.cuda()
        else:
            model = model

        labels, echoes = list(
            map(lambda x: to_variable(x, is_cuda=is_cuda), ds))

        echoes = echoes.float()
        outputs = model(echoes)
        _, preds = torch.max(outputs.data, 1)

        accuracies_per_class = plot_confusion_matrix(labels.cpu(), preds.cpu(), labels_in=[0, 1, 2, 5, 6, 7, 3, 4, 8, 9, 10, 11], epoch=38)
        print('bat', (accuracies_per_class[0]+accuracies_per_class[1]+accuracies_per_class[2]+accuracies_per_class[5]+accuracies_per_class[6]+accuracies_per_class[7])/6)
        print('notbat', (accuracies_per_class[3]+accuracies_per_class[4]+accuracies_per_class[8]+accuracies_per_class[9]+accuracies_per_class[10]+accuracies_per_class[11])/6)

        # accuracy
        total += labels.size(0)
        correct += (preds == labels).sum().item()

        accuracy += 100 * correct / total

    print('Test accuracy: %1d' %
          (accuracy / len(test_loader)))
