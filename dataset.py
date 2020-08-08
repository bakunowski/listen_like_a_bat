import torch
import pandas as pd
import numpy as np
import utils as u
import librosa
from scipy import signal
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

maxfilestoload = 10000
sr = 500000
# 180 deg / 101 measurments: each measurment is taken every 1.78... deg
angle_calc_multiplier = 1.7821782178217822

MAX_DB = -100
MIN_DB = -250


class EchoesDataset(Dataset):
    """ Bat echoes dataset:
        This instance will load a bat call convolved with one of the impulse
        responses of one flower - one echo. """

    def __init__(self, csv_file, bat_call_path, datapath, transform=None):
        """
        Args:
            csv_file (string): Path to csv file with annotations
            ir_csv (string): Path to data contaning flower impulse
                            responses.
            bat_call (audio signal): Audio signal of a bat call.
            datapath: path to folder with actual IRs in .csv format
            transform (callable, optional): Optional transform to be applied
                                            on a sample.
            size (int): how many data samples to load
        """
        self.plant_name = pd.read_csv(csv_file)
        self.folder = datapath
        self.transform = transform
        self.bat_call = self.get_bat_call(bat_call_path, plot=True)

    def __len__(self):
        index = self.plant_name.index
        return len(index)

    def load_data(self, path_to_csv):
        data = pd.read_csv(path_to_csv, sep=';', header=None)
        data = data.fillna(0)
        data = data.to_numpy()
        return data

    def get_echo(self, idx, plot=False):
        ir_csv = self.load_data(
            self.folder + '/' + self.plant_name.iloc[idx, 0])
        perseg = 256
        angle = idx * angle_calc_multiplier
        sample = []

        # get an ordered sequence of echoes from csv file for now
        for i in range(len(ir_csv)):
            echo = np.convolve(self.bat_call, ir_csv[i])
            f, t, spec = signal.spectrogram(echo, fs=500000,
                                            window='hann', nperseg=perseg,
                                            noverlap=perseg-200, detrend=False,
                                            scaling='spectrum')
            # spec_dB = 10 * np.ma.log10(spec)
            spec_norm = 10 * np.ma.log10(spec)
            #print("db: ", spec_dB[0])
            # spec_norm = u.normalize_0_1(spec_dB, MAX_DB, MIN_DB)
            #print("normalized db: ", spec_norm[0])
            #spec_dB = librosa.util.normalize(spec)

            if plot:
                plt.rcParams.update({'font.size': 12})
                plt.rcParams.update({'figure.dpi': 300})

                spec_dB = 10 * np.log10(spec)
                spec_min, spec_max = -100, -20

                plt.pcolormesh(t, f / 1000, spec_dB,
                               vmin=spec_min, vmax=spec_max)
                plt.colorbar()
                plt.ylabel('Frequency [kHz]')
                plt.xlabel('Time [s]')
                plt.show()

            sample.append(spec_norm)

        return np.expand_dims(sample, axis=1)

    def get_bat_call(self, path, plot=False):
        bat_call, _ = librosa.core.load(path, sr=sr)
        perseg = 256
        splits = librosa.effects.split(bat_call, top_db=15)
        sos = signal.butter(10, 8000, 'hp', fs=500000, output='sos')
        new_call = bat_call[splits[2][0]:splits[2][1]]
        new_call = signal.sosfilt(sos, new_call)

        if plot:
            f, t, spec = signal.spectrogram(new_call, fs=sr,
                                            window='hann', nperseg=perseg,
                                            noverlap=perseg-1, detrend=False,
                                            scaling='spectrum')

            spec_dB = 10 * np.log10(spec)
            spec_min, spec_max = -65, -15

            plt.pcolormesh(t, f / 1000, spec_dB, vmin=spec_min,
                           vmax=spec_max, cmap='magma')
            plt.show()

        return new_call

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = self.plant_name.iloc[idx, 1]
        echo = self.get_echo(idx, plot=False)

        return label, echo


# little helper to show one sample from current batch of echoes
def show_echo_batch(label, echoes):
    # print("Label: ", label)
    # print("Echoes: ", echoes.size())
    fig = plt.figure()
    x = 0

    for i in range(len(echoes)):
        for j in range(len(echoes[0])):
            x += 1
            ax = plt.subplot(len(echoes), len(echoes[0]), x)
            ax.set_title('Echo {echo}, Class {clas}'.format(
                echo=i+j+1, clas=label[i]))
            ax.axis('off')

            new_echo = echoes[:, j, 0]
            spec_dB = new_echo[i]
            # spec_min, spec_max = 0.8, 1
            spec_min, spec_max = -100, -20

            # im = plt.pcolormesh(spec_dB, vmin=spec_min,
            #                     vmax=spec_max, cmap='magma')
            im = plt.pcolormesh(spec_dB, cmap='magma')

    fig.suptitle(
        'One input sample to the network for classes {}'.format(label.numpy()), size=18)
    plt.show()

