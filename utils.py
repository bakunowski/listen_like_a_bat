import librosa, librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def calculate_spectrogram(y, window_size, hop_size):
    starts  = np.arange(0,len(y),window_size-hop_size,dtype=int)

    # centre frames (first frame centred at t=0)
    # add zeros of size half the frame to the beggining and end of audiofile
    y = np.concatenate([np.zeros(window_size//2), y, np.zeros(window_size//2)])
    # calculate total number of frames
    frame_count = int(np.floor(len(y) - window_size) / hop_size + 1)
    
    # derive some variables
    ffts = int(window_size / 2)
    _ = int(window_size / 2)  # bins - initial number equal to ffts, can change if filters are used
        
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
        plt.imshow(librosa.power_to_db(np.abs(D), ref=np.max), aspect='auto', origin='lower')
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

def retrieve_fingerprint(data, call, name):
    data = data.to_numpy()
    # n_fft = 65536
    n_fft = 131072
    hop_size = n_fft//32
    x = n_fft // 2
    y = 1 
    zeros = np.zeros((101, x, y))

    for i, ir in enumerate(data):
        # echo fingerprint
        zeros[i], _ = calculate_spectrogram(np.trim_zeros(ir), n_fft, hop_size)

        # bat call over fingerprint
        # echo = np.convolve(np.trim_zeros(ir), np.trim_zeros(call))
        # zeros[i], _ = calculate_spectrogram(np.trim_zeros(echo), n_fft, hop_size)
    
    spec = zeros.reshape(101, x*y)

    plt.imshow(librosa.power_to_db(np.abs(spec.T), ref=np.max),
                aspect='auto', origin='lower', cmap='plasma')
    # plt.imshow(librosa.power_to_db(np.abs(spec.T), ref=np.max),
    #             aspect='auto', origin='lower')
    # plt.show()
    plt.savefig('%s.png' % name, dpi=96)