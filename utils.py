import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import signal


def calculate_spectrogram(y, window_size, hop_size):
    starts = np.arange(0, len(y), window_size-hop_size, dtype=int)

    # centre frames (first frame centred at t=0)
    # add zeros of size half the frame to the beggining and end of audiofile
    y = np.concatenate([np.zeros(window_size//2), y, np.zeros(window_size//2)])
    # calculate total number of frames
    frame_count = int(np.floor(len(y) - window_size) / hop_size + 1)

    # derive some variables
    ffts = int(window_size / 2)
    # bins - initial number equal to ffts, can change if filters are used
    _ = int(window_size / 2)

    # init STFT matrix
    stft = np.empty([frame_count, ffts], np.complex)

    # create windowing function
    window = np.hanning(window_size)

    # step through all frames
    for frame in range(frame_count):
        start = frame * hop_size
        signal = y[start:start+window_size]
        # multiply the signal with the window function
        signal = signal * window
        # DFT
        stft[frame] = np.fft.fft(signal, window_size)[:ffts]
        # next frame
    # magnitude spectrogram
    spec = np.abs(stft)
    specX = 10*np.log10(spec)
    specX = specX.T

    return specX, starts


def return_bat_calls(call_datapath, sr=500000, plot=False):
    # the synthesised bat call (samling frequency = 500kHz)
    bat_call, sr = librosa.core.load(call_datapath, sr=sr)

    # naive separation of calls
    # TODO: change this!
    call1 = bat_call[2500:4000]
    call2 = bat_call[9500:11000]
    call3 = bat_call[18500:20000]

    if plot == True:
        plt.subplot(2, 1, 1)
        # call waveform
        librosa.display.waveplot(bat_call, sr=sr)

        plt.subplot(2, 1, 2)
        # spectrogram of a call
        D = librosa.stft(np.trim_zeros(bat_call), n_fft=64)
        plt.imshow(librosa.power_to_db(np.abs(D), ref=np.max),
                   aspect='auto', origin='lower')
        # plt.colorbar()
        plt.show()

    return call1, call2, call3


def load_data(path_to_csv, show_head=False):
    # changing the BOM NaN value to zero until a better solution is found
    data = pd.read_csv(path_to_csv, sep='\t', header=None)
    data = data.fillna(0)
    if show_head == True:
        print(data.head())

    return data


def get_kHz_scale_vec(ks, sample_rate, Npoints):
    freq_kHz = (ks*sample_rate/Npoints) // 1000
    freq_kHz = [int(i) for i in freq_kHz]
    return(freq_kHz)


def retrieve_fingerprint(data, call, name):
    data = data.to_numpy()
    n_fft = 4096
    hop_size = n_fft//2
    x = n_fft // 2
    y = 1
    zeros = np.zeros((101, x, y))

    for i, ir in enumerate(data):
        # echo fingerprint
        zeros[i], _ = calculate_spectrogram(ir, n_fft, hop_size)

    spec = zeros.reshape(101, x*y)

    plt.rcParams.update({'font.size': 12})

    ks = np.linspace(0, spec.shape[1], 15)
    ksHz = get_kHz_scale_vec(ks, 500000, n_fft*y)
    plt.yticks(ks, ksHz)
    plt.ylabel('Frequency (kHz)')

    t = np.linspace(0, 100, 7)
    my_ticks_x = [-90, -60, -30, 0, 30, 60, 90]
    plt.xticks(t, my_ticks_x)
    plt.xlabel('angle of sound incidence (°)')

    axes = plt.gca()
    axes.set_xlim([0, None])
    axes.set_ylim([280, 1320])

    plt.imshow(np.abs(spec.T), aspect='auto',
               origin='lower')
    plt.show()
    # plt.savefig('%s.png' % name, dpi=96)

def get_convolved_call(data, call, name):
    data = data.to_numpy()
    perseg = 256
    starting_angle = -90

    for ir in data:
        # bat call over each ir
        echo = np.convolve(call, ir) 
        f, t, spec = signal.spectrogram(np.trim_zeros(echo), fs=500000, window='hann', nperseg=perseg,
                                        noverlap=perseg-1, detrend=False, scaling='spectrum')

        plt.rcParams.update({'font.size': 12})

        # plt.pcolormesh(t, f / 1000, spec, cmap='magma')
        plt.pcolormesh(t, f / 1000, 10 *  np.log10(spec))
        plt.ylabel('Frequency [kHz]')
        plt.xlabel('Time [s]')
        plt.show()
        # plt.savefig('%s_Glosso_%s' % (name, round(starting_angle)), dpi=96)
        # starting_angle += 1.8