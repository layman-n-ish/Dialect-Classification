import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import librosa as lr
import librosa.display
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix    
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb

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

def plot_confusion_matrix(conf_matrix, title):
    df_cm = pd.DataFrame(conf_matrix, range(9), range(9))
    #print(df_cm)
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})
    print("\nPlotted the confusion matrix!\n")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Model: '+title)
    plt.show()

def feature_engineering(audio, Fs):
    features = []
    
    mfcc_coeff = mfcc(audio, Fs, winlen=0.023)
    fbank_coeff = logfbank(audio, Fs, winlen=0.023)
    delta_mfcc_coeff = delta(mfcc_coeff, 2)
    chroma_coeff = lr.feature.chroma_stft(y=audio, sr=Fs, n_fft=512)
    melspectro_coeff = lr.feature.melspectrogram(y=audio, sr=Fs, n_fft=512, n_mels=40)
    rmse_coeff = lr.feature.rmse(y=audio, frame_length=512)

    mfcc_mean = np.mean(mfcc_coeff, axis=0)
    mfcc_var = np.var(mfcc_coeff, axis=0)
    fbank_mean = np.mean(fbank_coeff, axis=0)
    fbank_var = np.var(fbank_coeff, axis=0)
    delta_mean = np.mean(delta_mfcc_coeff, axis=0)
    delta_var = np.var(delta_mfcc_coeff, axis=0)
    chroma_mean = np.mean(chroma_coeff.T, axis=0)
    chroma_var = np.var(chroma_coeff.T, axis=0)
    melspectro_mean = np.mean(melspectro_coeff.T, axis=0)
    melspectro_var = np.var(melspectro_coeff.T, axis=0)
    rmse_mean = np.mean(rmse_coeff.T, axis=0)
    rmse_var = np.var(rmse_coeff.T, axis=0)

    features.append(mfcc_mean)
    features.append(mfcc_var )
    features.append(fbank_mean)
    features.append(fbank_var)
    features.append(delta_mean)
    features.append(delta_var)
    features.append(chroma_mean)
    features.append(chroma_var)
    features.append(melspectro_mean)
    features.append(melspectro_var)
    features.append(rmse_mean)
    features.append(rmse_var)

    flat_features = [feat for sublist in features for feat in sublist]
    
    return flat_features

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
    X.to_csv('X_new_2.csv')
    print("\nDone making the CSV!\n")

def train_model():
    data = pd.read_csv('X_new_2.csv')
    train, test = train_test_split(data, test_size=0.3, random_state=2, shuffle=True)
    
    l = []
    for i in range(0, 210):
        l.append(str(i))

    X_train = train.loc[:, l]
    y_train = train.loc[:, '210']   
    X_test = test.loc[:, l]
    y_test = test.loc[:, '210']

    #print(X_train.head())
    #print(y_train.head())
    
    # train_lr(X_train, y_train, X_test, y_test)
    # train_svm(X_train, y_train, X_test, y_test)
    # train_knn(X_train, y_train, X_test, y_test)
    # train_nb(X_train, y_train, X_test, y_test)
    # train_rf(X_train, y_train, X_test, y_test)
    # train_xgb(X_train, y_train, X_test, y_test)
    train_adab(X_train, y_train, X_test, y_test)

def train_lr(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    model = LogisticRegression(C=3)
    print("\nLogistic Regression: Training...")
    model.fit(X_train, y_train)
    print("\nLogistic Regression: Training score: %f"%(model.score(X_train, y_train)))
    print("\nLogistic Regression: Test score: %f"%(model.score(X_test, y_test)))
    print("\n----------------------------------------------\n")
    y_pred = model.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(conf_mat, 'lr')

def train_svm(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    model = model = svm.SVC(kernel='linear')
    print("\nSupport Vector Machine: Training...")
    model.fit(X_train, y_train)
    print("\nSupport Vector Machine: Training score: %f"%(model.score(X_train, y_train)))
    print("\nSupport Vector Machine: Test score: %f"%(model.score(X_test, y_test)))
    print("\n----------------------------------------------\n")
    y_pred = model.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(conf_mat, 'svm')

def train_knn(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    model = KNeighborsClassifier()
    print("\nK Nearest Neighbor: Training...")
    model.fit(X_train, y_train)
    print("\nK Nearest Neighbor: Training score: %f"%(model.score(X_train, y_train)))
    print("\nK Nearest Neighbor: Test score: %f"%(model.score(X_test, y_test)))
    print("\n----------------------------------------------\n")
    y_pred = model.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(conf_mat, 'knn')

def train_nb(X_train, y_train, X_test, y_test):

    model = GaussianNB()
    print("\nGaussian Naive Bayes: Training...")
    model.fit(X_train, y_train)
    print("\nGaussian Naive Bayes: Training score: %f"%(model.score(X_train, y_train)))
    print("\nGaussian Naive Bayes: Test score: %f"%(model.score(X_test, y_test)))
    print("\n----------------------------------------------\n")
    y_pred = model.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(conf_mat, 'nb')

def train_rf(X_train, y_train, X_test, y_test):

    model = RandomForestClassifier(n_estimators=400, max_depth=20, min_samples_split=5, min_samples_leaf=5)
    print("\nRandom Forest: Training...")
    model.fit(X_train, y_train)
    print("\nRandom Forest: Training score: %f"%(model.score(X_train, y_train)))
    print("\nRandom Forest: Test score: %f"%(model.score(X_test, y_test)))
    print("\n----------------------------------------------\n")
    y_pred = model.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(conf_mat, 'rf')

def train_xgb(X_train, y_train, X_test, y_test):

    model = xgb.XGBClassifier(n_estimators=200, learning_rate=0.5)
    print("\nXGBoost: Training...")
    model.fit(X_train, y_train)
    print("\nXGBoost: Training score: %f"%(model.score(X_train, y_train)))
    print("\nXGBoost: Test score: %f"%(model.score(X_test, y_test)))
    print("\n----------------------------------------------\n")
    y_pred = model.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(conf_mat, 'xgb')

def train_adab(X_train, y_train, X_test, y_test):
    
    model = AdaBoostClassifier(n_estimators=700, learning_rate=0.01)
    print("\nAdaboost: Training...")
    model.fit(X_train, y_train)
    print("\nAdaboost: Training score: %f"%(model.score(X_train, y_train)))
    print("\nAdaboost: Test score: %f"%(model.score(X_test, y_test)))
    print("\n----------------------------------------------\n")
    y_pred = model.predict(X_test)
    conf_mat = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(conf_mat, 'adab')
    
if __name__ == "__main__":
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    train_model()
  