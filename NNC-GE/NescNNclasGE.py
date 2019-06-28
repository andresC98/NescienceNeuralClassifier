#Math, Data Science and System libraries
import numpy  as np
import pandas as pd
import math, collections
import time , sys, os
from random import randint
import matplotlib.pyplot as plt

# Compression algorithms
import bz2, lzma, zlib

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_digits
# Keras & tensorflow
import keras
from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential, clone_model
from keras import optimizers, losses
from keras.utils import to_categorical
from keras import backend as K

from queue import Queue
queue = Queue(10)

#Genetic Algorithm Library (PonyGE2)
from fitness.base_ff_classes.base_ff import base_ff

class NescNNclasGE(base_ff):
    maximise = False #we want to minimize our objective: Nescience value.
    global queue
    # Constants defintion
    INVALID_INACCURACY = -1    # If algorithm cannot be applied to this dataset
    INVALID_REDUNDANCY = -2    # Too small model    
    
    def __init__(self):

        super().__init__()
        #class attributes (placeholders, later as init attributes)
        niterations=25
        learning_rate=0.01
        method="Harmonic"
        compressor="bz2"
        backward=False
        verbose=True

        self.nu = [3]   # Start with one hidden layer with three units
        self.it         = niterations
        self.lr         = learning_rate
        self.method     = method
        self.compressor = compressor
        self.backward   = backward
        self.verbose    = verbose
        
        #this should be setted up here (?)
        data = load_digits()
        X = data.data
        X = (X - np.min(X)) / (np.max(X) - np.min(X))
        y = data.target
        self.X self.X_test, self.y, self.y_test = train_test_split(X, y, test_size=0.33)
        self.nsc  = None
        self.msd  = None
        self.viu  = None
        #Updated Nescience library variables
        self.lcdX = None
        self.lcdY = None
        self.norm_mscd = None
        self.tcc = None
        
        self.classes_  = None
        self.n_classes = None

    def evaluate(self, ind, **kwargs):

        model = eval(ind.phenotype)
        print("Layers being used: ", model.layers)
        

        model.compile()
        model.compile(loss = losses.categorical_crossentropy ,optimizer = sgd, metrics=['accuracy'])
        model.fit(x = msdX, y= self.y, validation_split=0.33,verbose=0,batch_size = 32, epochs = self.it)

        #Compute i
        nsc = self.nescience(msd, viu, model, msdX)

        return nsc #this will be the target to minimize by the GE algorithm.


    def nescience
