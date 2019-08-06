import subprocess, time
from keras.datasets import fashion_mnist
from sklearn.datasets import load_digits, load_iris
from imblearn.under_sampling import RandomUnderSampler

import pandas as pd
import numpy as np
"""
Contains all data to be used 

Except for the FashionMNIST load function, all of the other load functions should
have after their call this ttsplit so that X_reserved and y_reserved are NOT used in 
the algorithm and only outside for test evaluation of the model resulted.

#X, X_reserved, y, y_reserved  = train_test_split(X,y, random_state = 42,test_size = 0.33)

"""
#---------------------------------------------------
#Imbalanced datasets [Pulsar Stars, Magic Telescope]
#---------------------------------------------------

def load_pulsars():

    data = pd.read_csv("./datasets/pulsar_stars.csv")
    y=data.target_class
    X=data.drop("target_class", axis=1).values

    return X, y

def load_magic():

    data = pd.read_csv("./datasets/magic04.data", names=["fLength", "fWidth", "fSize", "fConc","fConc1","fAsym","fM3Long","fM3Trans","fAlpha","fDist","class"])
    y = (data['class'] == 'h').astype(int)
    X = data.drop('class', 1)

    return X, y

#Random undersampling of the majority class for Imb. datasets (optional)
# rus = RandomUnderSampler()
# X, y = rus.fit_resample(X,y)
# print("Resampled data shapes: {}, {}.".format(X.shape, y.shape))

#---------------------------------
#Balanced datasets [Digits, Iris]
#----------------------------------

def load_digitsdata():
    data = load_digits()
    X = data.data
    y = data.target

    return X, y

def load_irisdata():

    data = load_iris()
    X = data.data
    y = data.target

    return X, y

#Fashion MNIST 

def load_fashionmnist():

    ((X, y), (X_reserved, y_reserved)) = fashion_mnist.load_data()
    X = X.reshape(X.shape[0],X.shape[1]*X.shape[2]) #flatten array

    return X, y


def launch_data():
    print("Loading data...")
    with open("usedata.txt") as f: 
        dataset = f.readlines()[0]
        if "iris" in dataset:
            data =  load_irisdata()
        elif "digits" in dataset:
            data = load_digitsdata()
        elif "magic" in dataset:
            data = load_magic()
        elif "pulsars" in dataset:
            data = load_pulsars()
        elif "fmnist" in dataset:
            data = load_fashionmnist()
        else:
            print("Unknown data selected")
            exit(1)
        
        time.sleep(2)
        print("Done! Loaded {} dataset".format(dataset))

    return data[0], data[1], dataset


