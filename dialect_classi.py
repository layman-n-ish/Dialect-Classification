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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

import warnings

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
    features = []
    
    mfcc_coeff = mfcc(audio, Fs, winlen=0.023)
    fbank_coeff = logfbank(audio, Fs, winlen=0.023)
    delta_mfcc_coeff = delta(mfcc_coeff, 2)

    tot_frames = mfcc_coeff.shape[0]
    mfcc_coeffs_len = mfcc_coeff.shape[1]
    fbank_coeffs_len = fbank_coeff.shape[1]

    for j in range(mfcc_coeffs_len):
        mfcc_coeffs = []
        delta_coeffs = []
        for i in range(tot_frames):
            mfcc_coeffs.append(mfcc_coeff[i][j])
            delta_coeffs.append(delta_mfcc_coeff[i][j])
        features.append(np.mean(mfcc_coeffs))
        features.append(np.var(mfcc_coeffs))
        features.append(np.mean(delta_coeffs))
        features.append(np.var(delta_coeffs))

    for j in range(fbank_coeffs_len):
        fbank_coeffs = []
        for i in range(tot_frames):
            fbank_coeffs.append(fbank_coeff[i][j])    
        features.append(np.mean(fbank_coeffs))
        features.append(np.var(fbank_coeffs))

    return features

def get_audio_files(dialect_class):
    audio_files = glob.glob("../Read_Up/%s/*" %(dialect_class))

    return audio_files

def prep_data():
    data = {}

    i = 0
    for dialect_class in dialects:
        print("\nPrepping data for class %s" %(dialect_class))
        audio_files = get_audio_files(dialect_class)

        for file in audio_files:
            aud, Fs = lr.load(file)
            feature_arr = feature_engineering(aud, Fs)
            feature_arr.append(dialects.index(dialect_class))
            data[i] = feature_arr
            i = i+1

    X = pd.DataFrame.from_dict(data, orient='index')
    X.to_csv('X_new.csv')
    print("\nDone making the CSV!\n")

def train_model():
    data = pd.read_csv('X_new.csv')
    train, test = train_test_split(data, test_size=0.33, random_state=1, shuffle=True)
    
    l = []
    for i in range(0, 104):
        l.append(str(i))

    X_train = train.loc[:, l]
    y_train = train.loc[:, '104']   
    X_test = test.loc[:, l]
    y_test = test.loc[:, '104']

    #print(X_train.head())
    #print(y_train.head())
    
    train_rf(X_train, y_train, X_test, y_test)
    train_xgb(X_train, y_train, X_test, y_test)
    train_lr(X_train, y_train, X_test, y_test)

def train_lr(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    model = LogisticRegression(C=2, solver='newton-cg', n_jobs=-1, tol=0.01)
    print("\nLogistic Regression: Training...")
    model.fit(X_train, y_train)
    print("\nLogistic Regression: Training score: %f"%(model.score(X_train, y_train)))
    print("\nLogistic Regression: Test score: %f"%(model.score(X_test, y_test)))

def train_rf(X_train, y_train, X_test, y_test):

    model = RandomForestClassifier(n_estimators=400, max_depth=20, min_samples_split=5, min_samples_leaf=5)
    print("\nRandom Forest: Training...")
    model.fit(X_train, y_train)
    print("\nRandom Forest: Training score: %f"%(model.score(X_train, y_train)))
    print("\nRandom Forest: Test score: %f"%(model.score(X_test, y_test)))

def train_xgb(X_train, y_train, X_test, y_test):

    model = xgb.XGBClassifier(n_estimators=200, learning_rate=0.5)
    print("\nXGBoost: Training...")
    model.fit(X_train, y_train)
    print("\nXGBoost: Training score: %f"%(model.score(X_train, y_train)))
    print("\nXGBoost: Test score: %f"%(model.score(X_test, y_test)))

    
if __name__ == "__main__":
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    train_model()
  