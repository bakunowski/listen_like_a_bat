import torch
import pandas as pd
import numpy as np
import librosa
from scipy import signal
from torch.utils.data import Dataset, DataLoader, random_split
from random import randint
import matplotlib.pyplot as plt

# Csv file with folder/name of audio file and class
labelscv = "/home/bakunowski/Documents/QueenMary/FinalProject/listen_like_a_bat/listen_like_a_bat.csv"

# Path to folder with folders containing .csv files with impulse responses
folder = "/home/bakunowski/Documents/QueenMary/FinalProject/listen_like_a_bat/Data"

# Path to bat call
call_datapath = "/home/bakunowski/Documents/QueenMary/FinalProject/listen_like_a_bat/FlowerClassification/EcholocationCalls/Glosso_call.wav"

maxfilestoload = 10000
sr = 500000
# 180 deg / 101 measurments: each measurment is taken every 1.78... deg
angle_calc_multiplier = 1.7821782178217822


class Echoes(Dataset):
    """ Bat echoes dataset:
        This instance will load a bat call convolved with one of the impulse
        responses of one flower - one echo. """

    def __init__(self, csv_file, bat_call_path, size, transform=None):
        """
        Args:
            csv_file (string): Path to csv file with annotations
            ir_csv (string): Path to data contaning flower impulse
                            responses.
            bat_call (audio signal): Audio signal of a bat call.
            transform (callable, optional): Optional transform to be applied
                                            on a sample.
            size (int): how many data samples to load
        """
        self.plant_name = pd.read_csv(csv_file)
        self.transform = transform
        self.size = size
        self.bat_call = self.get_bat_call(bat_call_path, plot=False)

    def __len__(self):
        index = self.plant_name.index
        return len(index)

    def load_data(self, path_to_csv):
        data = pd.read_csv(path_to_csv, sep=';', header=None)
        data = data.fillna(0)
        data = data.to_numpy()
        return data

    def get_echo(self, idx, plot=False):
        ir_csv = self.load_data(folder + '/' + self.plant_name.iloc[idx, 0])
        perseg = 256
        angle = idx * angle_calc_multiplier

        # get a random echo from csv file for now
        echo = np.convolve(self.bat_call, ir_csv[0])
        f, t, spec = signal.spectrogram(echo, fs=500000,
                                        window='hann', nperseg=perseg,
                                        noverlap=perseg-5, detrend=False,
                                        scaling='spectrum')

        if plot:
            plt.rcParams.update({'font.size': 12})
            plt.rcParams.update({'figure.dpi': 300})

            spec_dB = 10 * np.log10(spec)
            spec_min, spec_max = -100, -20

            plt.pcolormesh(t, f / 1000, spec_dB, vmin=spec_min, vmax=spec_max)
            plt.colorbar()
            plt.ylabel('Frequency [kHz]')
            plt.xlabel('Time [s]')
            plt.show()

        return spec

    def get_bat_call(self, path, plot=False):
        bat_call, _ = librosa.core.load(path, sr=sr)
        perseg = 256
        splits = librosa.effects.split(bat_call, top_db=15)
        new_call = bat_call[splits[0][0]:splits[0][1]]

        if plot:
            f, t, spec = signal.spectrogram(new_call, fs=sr,
                                            window='hann', nperseg=perseg,
                                            noverlap=perseg-20, detrend=False,
                                            scaling='spectrum')

            spec_dB = 10 * np.log10(spec)
            spec_min, spec_max = -65, -15

            plt.pcolormesh(t, f / 1000, spec_dB, vmin=spec_min, vmax=spec_max)
            plt.show()

        return new_call

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = self.plant_name.iloc[idx, 1]
        echo = self.get_echo(idx, plot=False)

        return label, echo


# little helper to show current batch of echoes
def show_echo_batch(label, echoes):
    fig = plt.figure()

    for i in range(len(echoes)):

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')

        spec_dB = 10 * np.log10(echoes[i])
        spec_min, spec_max = -100, -20

        plt.pcolormesh(spec_dB, vmin=spec_min, vmax=spec_max)

    plt.show()


echoes_dataset = Echoes(labelscv, call_datapath, maxfilestoload)

trainset, valset = random_split(echoes_dataset, [200, 34])

train_loader = DataLoader(trainset, batch_size=4, shuffle=True)
val_loader = DataLoader(valset, batch_size=4, shuffle=True)

for i_batch, sample_batched in enumerate(train_loader):
    print(i_batch, sample_batched[1].size(), sample_batched[0].size())
    # show_echo_batch(sample_batched[0], sample_batched[1])
