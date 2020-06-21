import torch
import pandas as pd
import numpy as np
import librosa
from scipy import signal
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
# TODO:
# get rid of NaN values DONE

# one hot encoding of classes path:
# perhaps as a csv file with name of audio file and encoding?

# path to IR file
# TODO: this only loads file for one flower (one measurment)
# extend to load for different flowers with batches etc!!!!!
ir_path = "/home/bakunowski/Documents/QueenMary/FinalProject/listen_like_a_bat/FlowerClassification/FlowerIRs/Burmeist_h_impuls/Burmeist h I 20cm_2009-06-29_NA_2019-08-11_impuls.csv"

# path to bat call
call_datapath = "/home/bakunowski/Documents/QueenMary/FinalProject/listen_like_a_bat/FlowerClassification/EcholocationCalls/Glosso_call.wav"

maxfilestoload = 100
sr = 500000


class Echoes(Dataset):
    """ Bat echoes dataset:
        This instance will load a bat call convolved with one of the impulse
        responses of one flower - one echo. """

    def __init__(self, ir_csv_file, bat_call_path, size, transform=None):
        # def __init__(self, csv_file, ir_csv, bat_call, size, transform=None):
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
        # self.plant_name = pd.read_csv(csv_file)
        self.transform = transform
        self.size = size
        # self.ir_csv = pd.read_csv(ir_csv_file, sep='\t', header=None)
        self.ir_csv = self.load_data(ir_csv_file)
        self.bat_call = self.get_bat_call(bat_call_path, plot=False)
        self.echo = []

    def __len__(self):
        return self.size

    def load_data(self, path_to_csv):
        data = pd.read_csv(path_to_csv, sep='\t', header=None)
        data = data.fillna(0)
        return data

    def get_echo(self, idx, plot=False):
        ir_csv = self.ir_csv.to_numpy()
        perseg = 256
        # TODO: how to calculate this angle properly?
        # starting_angle = -90

        echo = np.convolve(self.bat_call, ir_csv[idx])
        f, t, spec = signal.spectrogram((echo), fs=500000,
                                        window='hann', nperseg=perseg,
                                        noverlap=perseg-1, detrend=False,
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
            plt.close()
        return new_call

    # currently getitem fetches an echo from one csv file of one flower
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # TODO:
        # label =
        print(idx)
        echo = self.get_echo(idx, plot=True)
        return echo


echoes_dataset = Echoes(ir_path, call_datapath, 10)
print(len(echoes_dataset))

test = echoes_dataset[1]
print(test.shape)

train_loader = DataLoader(echoes_dataset, batch_size=2, shuffle=True)

for i_batch, sample_batched in enumerate(train_loader):
    print(i_batch, sample_batched.size())
