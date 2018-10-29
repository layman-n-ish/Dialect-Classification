import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa as lr
import librosa.display
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank

dialects = ["IDR1", "IDR2", "IDR3", "IDR4", "IDR5", 
"IDR6", "IDR7", "IDR8", "IDR9"]

def audio_plot(audio_file):
    audio, Fs = lr.load(audio_file) #Fs is the sampling rate

    #print(audio.size/Fs) #gives the time duration of the audio signal
    #print(Fs)

    Ts = 1.0/Fs #Time step of a single sample
    t = np.arange(0, len(audio)/Fs, Ts) #time vector

    N = len(audio) #total number of samples
    #k = np.arange(N)
    #T = N/Fs #time duration of the audio signal
    #f = k/T #since I need total 'k' samples in the time duration of the audio
    f = np.fft.fftfreq(N, d=Ts)

    fft_audio = np.fft.fft(audio)/N

    fig, ax = plt.subplots(2, 1, figsize=(12, 4))
    ax[0].plot(t, audio)
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Amplitude')
    
    ax[1].plot(f/1000, abs(fft_audio)) 
    ax[1].set_xlabel('Freq (KHz)')
    ax[1].set_ylabel('|Y(freq)|')

def feature_engineering(audio, Fs):
    features = {}

    #stft
    #mfcc
    #chroma vector, deviation
    #spectral_contrast
    #tonnetz
    #zero crossing rate
    #spectral centroid
    #spectral flux

    return features

def get_audio_files(dialect_class):
    audio_files = glob.glob("../Read_Up/%s/*" %(dialect_class))

    return audio_files

def prep_data():
    data = []
    labels = []

    for dialect_class in dialects:
        print("Prepping data for class %s" %(dialect_class))
        audio_files = get_audio_files(dialect_class)

        for file in audio_files:
            aud, Fs = lr.load(file)
            feature_dict = feature_engineering(aud, Fs)

            data.append(feature_dict)
            label.append(dialects.index(dialect_class))

    return data, labels

#def train_model():
    