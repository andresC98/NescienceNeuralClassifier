#@title

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#########################################################
# Nescience Neural Network Classifier Notebook version  #
#########################################################
# Test version v2 - Implementing New Nescience library  #
#-------------------------------------------------------#


"""

Building optimal neural networks based on the minimum nescience principle

@author: Rafael Garcia Leiva
@mail:   rgarcialeiva@gmail.com
@web:    http://www.mathematicsunknown.com/
@copyright: All rights reserved
@version: 0.2 (25 Jun, 2019)

The following internal attributes will be used

    * X          - predictive features
    * y          - target variable
    
    * nu         - number of units per hidden layer
      
    * ni         - number of iterations
    * lr         - learning rate

    * method     - method to copute nescience
    * compressor - compression algorithm used to compute redundancy
    * verbose    - print debugging information

"""

import math
import numpy  as np
import pandas as pd
import math, collections
import time , sys, os, logging
from random import randint
import matplotlib.pyplot as plt

# Compression algorithms
import bz2
import lzma
import zlib

# Scikit-learn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

# Keras & tensorflow
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model, Sequential, clone_model
from keras import optimizers, losses
from keras.utils import to_categorical
from keras import backend as K

from queue import Queue

queue = Queue(10)
import os, logging

class NescienceNeuralNetworkClassifier(BaseEstimator, ClassifierMixin):
    
    global queue
    
    # Constants defintion
    
    INVALID_INACCURACY = -1    # If algorithm cannot be applied to this dataset
    INVALID_REDUNDANCY = -2    # Too small model    
    
    def __init__(self, niterations=25, learning_rate=0.01, method="Harmonic", compressor="bz2", backward=False, verbose=False):
        """
        Initialization of the model
    
          * method:  String. Select the method to compute the nescience of the
                     tree, valid values are "Euclid", "Harmonic", "Geometric",
                     "Entropy" and "Inner"
          * niterations: Int. Maximum number of iterations used during the
                         training of the neural network
          * verbose:     Boolean. If true, prints out additional information
        """
       
        # TODO: check the input parameters
        
        self.nu = [3]   # Start with one hidden layer with three units

        self.it         = niterations
        self.lr         = learning_rate

        self.method     = method
        self.compressor = compressor
        self.backward   = backward
        self.verbose    = verbose
        
        self.X    = None
        self.y    = None
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

        self.tol   = 0.05 #originally at 0.05
        self.decay = 0.05 #originally at 0.1

        self.history = None #Records history of Stats for each algorithm' saved NN
        self.model_hist = None #dataframe with history

    def fit(self, X, y, run_until = 0):
        """
        Fit a model (a neural network with a hidden layer) given a dataset
    
        The input dataset has to be in the following format:
    
           * X = list([[x11, x12, x13, ...], ..., [xn1, xn2, ..., xnm]])
           * y = list([y1, ..., yn]) 
       
        Return the fitted model
        """
        #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        logging.disable(logging.WARNING)
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

        # TODO: check the input parameters [DONE ?]
        if(len(X.shape) != 2 or len(y.shape) != 1):
            if(len(X.shape) == 3):
                X = X.reshape(X.shape[0],X.shape[1]*X.shape[1])
                print("New shape of X: {}".format(X.shape))
            else:   
                print("Invalid data shape/s of {} and output of {}.\nInput Array must be of format [[x11, x12, x13, ...], ..., [xn1, xn2, ..., xnm]].\nOutput array must be of format:[y1, ..., yn]".format(X.shape, y.shape))
                return
                #TODO Exit program?

        # TODO: test_size should be large enought, not a percentage of data
        #(validation splits will be done in each .fit automatically.)
        #self.X, self.X_test, self.y, self.y_test = train_test_split(X, y, test_size=0.3)

        #self.X      = np.array(self.X)
        #self.X_test = np.array(self.X_test)
        #self.y      = np.array(self.y)    
        #self.y_test = np.array(self.y_test)
        
        best_nsc = 1 #holds the best nsc achieved so far by the algorithm
        best_config = None #holds the NN associated with that nescience, its vius and msd

        self.X = np.array(X)
        #self.y = np.array(y)
        
        #self.classes_  = np.unique(y)
        self.classes_, self.y = np.unique(y, return_inverse=True)
        self.n_classes = self.classes_.shape[0]

        self.history = [] #no need to be a synchr. queue as it wont be used for real-time graphs 

        #time metrics
        start_time = time.time()

        #Computing optimal code lengths for the attributes and target
        self.lcdY = self._codelength_discrete(self.y)
    
        self.lcdX = np.zeros(self.X.shape[1])

        for i in np.arange(self.X.shape[1]):
            self.lcdX[i] = self._codelength_continuous(self.X[:,i])
        
        # Compute the contribution of each feature to miscoding
        
        #self._initmiscod() #old method
        self.tcc = self._tcc()
        self.norm_mscd = 1 - np.array(self.tcc)
        self.norm_mscd = self.norm_mscd / np.sum(self.norm_mscd)
        self.msd = self.norm_mscd.copy() #for compatibility with rest of the code (check if valid)
        #print("Dataset normalized miscoding / target conditional complexity: {}".format(self.msd))

        if self.backward:
            self.viu = np.ones(self.X.shape[1], dtype=np.int)
        else:
            self.viu = np.zeros(self.X.shape[1], dtype=np.int)


        #Idea: check total # of features of dataset. Initial network will have input variables
        #   depending of this number of features, to avoid slow start & stuck.
        #init_ft_space = int(np.ceil(X.shape[1]/8)) #TODO replace with sqrt
        init_ft_space = int(np.sqrt(X.shape[1]))
        self.nu[0] = init_ft_space
        msd = self.msd.copy()    
        msd = 1-msd    
        print("Search will start with {} features out of {}.".format(init_ft_space, X.shape[1]))
    
        for i in np.arange(init_ft_space): 
                            
            if self.backward:
                msd[np.where(self.viu == 0)] = 0
            else:
                msd[np.where(self.viu)] = 1

            if self.backward:
                self.viu[np.where(msd == np.max(msd))] = 1
            else:            
                self.viu[np.where(msd == np.min(msd))] = 1

        msdX = self.X[:,np.where(self.viu)[0]]
        self.y = to_categorical(self.y) #to use with categorical cross entropy
        #self.y_test = to_categorical(self.y_test) #to use with categorical cross entropy
        print("Input dimension: {}".format(msdX.shape[1]))

        #optimizer(s) setup (only required once)
        
        #adam = optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        sgd = optimizers.SGD(lr = self.lr, momentum = 0.9,nesterov = True)
        
      
        # Create the initial neural network
        #  - two features - UPDATED WITH sqrt N total features
        #  - one hidden layer
        #  - three units
        
        #Initial NN construction
        self.nn = Sequential()
        self.nn.add(Dense(units = self.nu[0], activation='relu', input_dim=msdX.shape[1]))
        self.nn.add(Dense(units = self.n_classes, activation = 'softmax'))
        self.nn.compile(loss = losses.categorical_crossentropy ,optimizer = sgd, metrics=['accuracy'])
        self.nn.fit(x = msdX, y= self.y, validation_split=0.33,verbose=0,batch_size = 32, epochs = self.it)

        print("[DEBUG] Evaluating initial nn...")
        score = self.nn.evaluate(msdX, self.y)
        print("[DEBUG] Keras-Scores obtained: {}:{}, {}:{}.".format(self.nn.metrics_names[0],score[0],self.nn.metrics_names[1],score[1]))
        
        self.nsc = self._nescience(self.msd, self.viu, self.nn, msdX)
        best_nsc = self.nsc
        best_config = (self.nn, self.viu)

        if self.nsc == self.INVALID_INACCURACY:
            # There is anything more we can do with this dataset
            # since current model is already nearly perfect
            print("WARNING: Invalid Nescience")
            if self.verbose:
                print("Miscoding: ", self._miscoding(self.msd, self.viu), "Inaccuracy: ", self._inaccuracy(self.nn, msdX), "Redundancy: ", self._redundancy(self.nn), "Nescience: ", self._nescience(self.msd, self.viu, self.nn, msdX))
                print(self._cnn2str())
            return self
            
        if self.verbose:
            vals, queue = self._update_vals(msdX)
            print("Miscoding: ", vals["miscoding"], "Inaccuracy: ", vals["inaccuracy"], "Redundancy: ", vals["surfeit"], "Nescience: ", vals["nescience"])
        
        # While the nescience decreases
        decreased = True        
        iter = 0
        while (decreased):
            iter+=1
            print("Best nsc so far: {}\nRun #{}.".format(best_nsc,iter))
            if(run_until != 0 and iter > run_until):
                print("Stopped algorithm, max runs achieved. {} out of {}".format(iter,run_until+1))
                break
            decreased = False

            #
            # Test adding a new feature  
            #
            
            # Check if therer are still more variables to add
            if (self.backward and (np.sum(self.viu) != 0)) or \
               ((not self.backward) and (np.sum(self.viu) != self.viu.shape[0])):
            
                msd = self.msd.copy()
                viu = self.viu.copy()
            
                if self.backward:
                    msd[np.where(viu == 0)] = 1
                else:
                    msd[np.where(viu)] = 0 #to avoid adding an existing one

                if self.backward:
                    viu[np.where(msd == np.min(msd))] = 0
                else:
                    viu[np.where(msd == np.max(msd))] = 1

                msdX = self.X[:,np.where(viu)[0]]
                #New Keras NN creation
                print("[DEBUG] Testing adding a new feature to cnn. {} nº of features.".format(msdX.shape[1]))
                init_t = time.time()

                cnn = Sequential() #network has to be created from scratch, we cannot reuse weights
                cnn.add(Dense(units = self.nu[0], activation='relu', input_dim=msdX.shape[1])) #first layer after inputs
                for layer in np.arange(len(self.nu)-1):
                    cnn.add(Dense(units = self.nu[layer+1], activation = 'relu'))
                cnn.add(Dense(units = self.n_classes, activation = 'softmax'))
                
                cnn.compile(loss = losses.categorical_crossentropy ,optimizer = sgd, metrics=['accuracy'])
                cnn.fit(x = msdX, y= self.y,validation_split=0.33, verbose=0,batch_size = 32, epochs = self.it)

                network_comp_t = time.time() - init_t
                print("Took {:.2f}s. to create and fit test NN.".format(network_comp_t))
                
                nsc = self._nescience(msd, viu, cnn, msdX)
                #nsc = self._nescience(self.msd, viu, cnn, msdX)
                if(nsc == 0):
                    break
                if nsc == self.INVALID_INACCURACY:
                    # We cannot do anything more with this dataset
                    self.nsc = nsc
                    self.nn  = cnn
                    self.viu = viu

                    if self.verbose:
                        vals, queue = self._update_vals(msdX)                     
                        print("Warning: invalid inaccuracy")
                        print("Miscoding: ", vals["miscoding"], "Inaccuracy: ", vals["inaccuracy"], "Redundancy: ", vals["surfeit"], "Nescience: ", vals["nescience"])

                    break
            
                # Save data if nescience has been reduced                        
                if (nsc - self.tol) < self.nsc:
                    print("DEBUG] Nescience reduced - SAVED THIS CONFIGURATION.")
                    #cnn.summary()
                    if(nsc < best_nsc):
                        best_nsc = nsc
                        best_config = (cnn, viu)
                        #best_nn = cnn
                    decreased = True
                    self.nsc = nsc
                    self.nn   = cnn
                    self.viu  = viu
                    if self.verbose:
                        vals, queue = self._update_vals(msdX)
                        if self.backward:
                            print("Removed a feature. Var in use: ", np.sum(self.viu), " Nescience: ", nsc)
                        else:
                            print("Added new feature - Nescience: ", nsc)
                else:
                    pass
                    #K.clear_session()

            #
            # Test adding a new layer
            #
            
            nu = self.nu.copy()
            nu.append(3)
            # if(randint(0, 100) > 75):
            #     nu.append(2*int(np.ceil(np.mean(nu))))
            # else:
            #     nu.append(int(np.ceil(np.mean(nu))))
           
            print("[DEBUG] Testing adding a new layer... Current hidden layers: {}.".format(nu)) 

            msdX = self.X[:,np.where(self.viu)[0]]
            init_t = time.time()
            # cnn = Sequential()
            # cnn.add(Dense(units = nu[0], activation='relu', input_dim=msdX.shape[1]))
            # for k, units in enumerate(nu):
            #     if(k > 0): #we already added the first hidden layer
            #         cnn.add(Dense(units, activation = 'relu'))
            
            #cnn.add(Dense(units = self.n_classes, activation = 'softmax'))
            #reusing weights from previous saved network's layers.
            cnn = Sequential()
            for layer in self.nn.layers[:-1]: # just exclude last layer from copying
                cnn.add(layer)
            for layer in cnn.layers:
                layer.trainable = False #to reuse their weights
            cnn.add(Dense(units = nu[-1], activation = 'relu')) #adding the extra layer
            cnn.add(Dense(units = self.n_classes, activation = 'softmax'))

            cnn.compile(loss = losses.categorical_crossentropy ,optimizer = sgd, metrics=['accuracy'])
            cnn.fit(x = msdX, y= self.y, validation_split=0.33,verbose=0,batch_size = 32, epochs = self.it)
            
            network_comp_t = time.time() - init_t
            print("Took {:.2f}s. to create and fit test NN.".format(network_comp_t))

            nsc = self._nescience(self.msd, self.viu, cnn, msdX)
            if(nsc == 0):
                    break

            if nsc == self.INVALID_INACCURACY:
                    # We cannot do anything more with this dataset
                    self.nsc = nsc
                    self.nn  = cnn
                    self.nu  = nu

                    if self.verbose:
                        vals, queue = self._update_vals(msdX)
                        print("Warning: invalid inaccuracy")
                        print("Miscoding: ", vals["miscoding"], "Inaccuracy: ", vals["inaccuracy"], "Redundancy: ", vals["surfeit"], "Nescience: ", vals["nescience"])

                    break
            
            # Save data if nescience has been reduced                        
            if (nsc - self.tol) < self.nsc:
                print("DEBUG] SAVED THIS CONFIGURATION.")
                #cnn.summary()
                if(nsc < best_nsc):
                    best_nsc = nsc
                    best_config = (cnn, self.viu)
                    #best_nn = cnn
                self.nsc  = nsc
                self.nn   = cnn
                self.nu   = nu
                decreased = True
                
                if self.verbose:
                    vals, queue = self._update_vals(msdX)
                    print("Added new layer - Nescience: ", nsc)
            else:
                pass
                #K.clear_session()


            for i in np.arange(len(self.nu)): #loops k times (in a NN with K hidden layers)
                nu = self.nu.copy()
                # if(randint(0, 100) > 75):
                #     nu[i] = 2*nu[i]
                # else:
                #     nu[i] = nu[i] + 1 #extra node added to hidden layer i     
                nu[i] = nu[i] + 1 #extra node added to hidden layer i     
                print("[DEBUG] Testing adding a new unit... Hidden layers: {}".format(nu))            
                msdX = self.X[:,np.where(self.viu)[0]]
                
                init_t = time.time() 
                cnn = Sequential()
                cnn.add(Dense(units = nu[0], activation='relu', input_dim=msdX.shape[1]))
                for k in np.arange(len(self.nu)-1):
                    cnn.add(Dense(units = nu[k+1], activation='relu'))
                cnn.add(Dense(units = self.n_classes, activation = 'softmax'))
                
                cnn.compile(loss = losses.categorical_crossentropy ,optimizer = sgd, metrics=['accuracy'])
                cnn.fit(x = msdX, y= self.y, validation_split=0.33,verbose=0,batch_size = 32, epochs = self.it)

                network_comp_t = time.time() - init_t
                print("Took {:.2f}s. to create and fit test NN.".format(network_comp_t))

                nsc = self._nescience(self.msd, self.viu, cnn, msdX)
                if(nsc == 0):
                    break
                if nsc == self.INVALID_INACCURACY:
                    # We cannot do anything more with this dataset
                    self.nsc = nsc
                    self.nu  = nu
                    self.nn  = cnn
                    # queue.put(self.nsc)
                    # TODO: Solve this problem
                    print("Warning! Panic! Warning! Panic! ...")
                    break
            
                # Save data if nescience has been reduced                        
                if (nsc - self.tol) < self.nsc: 
                    print("DEBUG] SAVED THIS CONFIGURATION.")
                    #cnn.summary()
                    if(nsc < best_nsc):
                        best_nsc = nsc
                        best_config = (cnn, self.viu)
                        # best_nn = cnn
                    self.nsc  = nsc
                    self.nn   = cnn
                    self.nu   = nu
                    decreased = True

                    if self.verbose:
                        vals, queue = self._update_vals(msdX)
                        print("Added new unit - Nescience: ", nsc)
                        print("[DEBUG] Nescience has been reduced after adding new unit")
                else:
                    pass
                    #K.clear_session()

        
            # Update tolerance
            self.tol = self.tol * (1 - self.decay)
            print("Tolerance: " + str(self.tol))


        # -> end while
        self.nn, self.viu = best_config #unpack saved best record.
        #self.nn = best_nn
        # Print out the best nescience achieved
        if self.verbose:
            
            if self.nsc == self.INVALID_INACCURACY:
                print("WARNING: Invalid Nescience")
            else:
                msdX = self.X[:,np.where(self.viu)[0]]
                print("Final -> Miscoding: ", self._miscoding(self.msd, self.viu), "Inaccuracy: ", self._inaccuracy(self.nn, msdX), "Redundancy: ", self._redundancy(self.nn), "Nescience: ", self._nescience(self.msd, self.viu, self.nn, msdX))

            print(self._nn2str(self.nn))
            self.nn.summary()
            elapsed_time = time.time() - start_time
            print("Total elapsed time for obtaining final network: {} s.".format(elapsed_time))
        
        print("Proceeding to test obtained network:")

        final_score = self.nn.evaluate(X[:,np.where(self.viu)[0]],self.y)
        # final_score = self.score(X[:,np.where(self.viu)[0]],self.y)

        print("Obtained final score of: {}.".format(final_score))

        time.sleep(2) #to make sure to log. 

        return self
 
    
    def predict(self, nn, X):
        """
        Predict class given a dataset
    
          * X = list([[x11, x12, x13, ...], ..., [xn1, xn2, ..., xnm]])
    
        Return a list of classes predicted
        """
        # TODO: Check that we have a model trained
        predictions = nn.predict(X) # Softmax output
        predictions = np.argmax(predictions, axis=1)
        #print("[DEBUG] Predictions after argmax interpretation: {}".format(predictions))
        return predictions


    def score(self, X, y):
        """
        Evaluate the performance of the current model given a test dataset

           * X = list([[x11, x12, x13, ...], ..., [xn1, xn2, ..., xnm]])
           * y = list([y1, ..., yn])
    
        Return one minus the mean error
        """
        
        # TODO: Check that we have a model trained>
        #Keras returns accuracy (mean accuracy). Is the same? (1-mean error === mean acc.)
        print("Input data shape: {}, Output data shape: {}.".format(X.shape, y.shape))
        # print("Neural Network to test:")
        # self.nn.summary()
        score = self.nn.evaluate(X, y, verbose=1)[1] #evaluate returns (loss, accuracy) tuple
        print("Obtained Network Evaluation Score of: {}".format(score))
        
        return score      

    
    """
    Compute the miscoding of the dataset used by the current model
      
    Return the miscoding
    """

    def _miscoding(self, msd, viu):
        miscoding = np.dot(viu, self.norm_mscd)
        #miscoding = 1 - miscoding
        #print("Miscoding obtained of: {}".format(miscoding))

        return miscoding
          
    
    """
    Computes the inaccuracy of the model
        
    Returns the inaccuracy (float)
    """
    #updated and modified to work with current Nescience library
    def _inaccuracy(self, nn, X):
                        
        # Compute the list of errors
        ldm = 0
        ldata = 0

        pred = self.predict(nn, X)

        errors = list()
        for i in np.arange(X.shape[0]):
            if(pred[i] != np.argmax(self.y, axis=1)[i]):
                new_error = list(X[i])
                new_error.append(np.argmax(self.y, axis=1)[i])
                errors.append(new_error)
        
        errors = np.array(errors)
        
        # Compute the compressed length of errors    
        for i in np.arange(len(errors[0])-1):
            ldm = ldm + self._codelength_continuous(errors[:,i])
        
        ldm = ldm + self._codelength_discrete(errors[:,-1].astype(np.int))
        # Compute the compressed length of data in use
            
        for i in np.arange(self.X.shape[1]):

            if self.viu[i] == 0:
                continue

            ldata = ldata + self.lcdX[i]
        
        ldata = ldata + self.lcdY
            
        inaccuracy = ldm / ldata
        
        return inaccuracy

    """
    Compute the redundancy of the current model (string-based representation)
    
    Warning: If the redundancy is a negative number it should not be used
         
    Return the redundancy (float) of the model.
    """
    def _redundancy(self, nn):
    
        # Compute the model string and its compressed version
        model = self._nn2str(nn).encode()
        
        if self.compressor == "lzma":
            compressed = lzma.compress(model, preset=9)
        elif self.compressor == "zlib": 
            compressed = zlib.compress(model, level=9)
        else: # By default use bz2   
            compressed = bz2.compress(model, compresslevel=9)
        
        # Check if the model is too small to compress
        if len(compressed) > len(model):
            return self.INVALID_REDUNDANCY
        
        # redundancy = 1 - l(m*) / l(m)
        redundancy = 1 - len(compressed) / len(model)
    
        return redundancy


    """
    Compute the nescience of a model,
    using the method specified by the user
    
    Warning: If the nescience is a negative number it should not be used
          
    Return nescience
    """
    def _nescience(self, msd, viu, nn, X):

        miscoding  = self._miscoding(msd, viu)
        redundancy = self._redundancy(nn)
        inaccuracy = self._inaccuracy(nn, X)

        if inaccuracy == self.INVALID_INACCURACY:
            # The inaccuracy is too small, there is anything
            # we can do with this dataset
            return self.INVALID_INACCURACY

        if redundancy < inaccuracy:
            # The model is still too small to compute the nescience
            # use innacuracy instead
            redundancy = 1 
    
        # Compute the nescience according to the method specified by the user
        if self.method == "Euclid":
            # Euclidean distance
            nescience = math.sqrt(miscoding**2 + inaccuracy**2 + redundancy**2)
        elif self.method == "Arithmetic":
            # Arithmetic mean
            nescience = (miscoding + inaccuracy + redundancy) / 3
        elif self.method == "Geometric":
            # Geometric mean
            nescience = np.pow(miscoding * inaccuracy * redundancy, 1/3)
        elif self.method == "Product":
            # The product of both quantities
            nescience = miscoding * inaccuracy * redundancy
        elif self.method == "Addition":
            # The product of both quantities
            nescience = miscoding + inaccuracy + redundancy            
        else:
            # By default use the Harmonic mean
            if inaccuracy == 0:
                # Avoid dividing by zero
                inaccuracy = np.finfo(np.float32).tiny 
            nescience = 3 / ( (1/miscoding) + (1/inaccuracy) + (1/redundancy))            
    
        return nescience

    """
    Compute accuracy of the current model over a test dataset
      
    Return the accuracy
    """
    def _score(self, nn, viu):

        #x_test = self.X_test[:,np.where(viu)[0]]
        x = self.X[:,np.where(viu)[0]]
        score = self.nn.evaluate(x, self.y)[1]
        print("[DEBUG] NN Evaluated Accuracy = {}.".format(score))
        return score


    """
    Encode the candidate neural networks as a string
    
    Return the string
    """
    def _nn2str(self, nn):
        # TODO: Review 
        # 10/06/19: Solved incorrect index printing of A_i, Z_i, W_i
        # 12/06/19: Migrated nn2str code for Keras model compatibility
        # Header
        string = "def NN(X):\n"
            
        # Parameters [for each layer i in network...]r

        for i in np.arange(len(nn.layers)):
            string = string + "    W" + str(i) + " = " + str(nn.layers[i].get_weights()[0]) + "\n"
            string = string + "    b" + str(i) + " = " + str(nn.layers[i].get_weights()[1]) + "\n"
            
        # Computation
        for i in np.arange(len(nn.layers)):               
            if(i==0):
                string = string + "    Z" + str(i) + " = np.matmul(W" + str(i) + ", X) + b" + str(i) + "\n"
                string = string + "    A" + str(i) + " = np.relu(Z" + str(i) + ")\n"
            else:
                string = string + "    Z" + str(i) + " = np.matmul(W" + str(i) + ", A" +str(i-1)+") + b" + str(i) + "\n"
                string = string + "    A" + str(i) + " = np.relu(Z" + str(i) + ")\n"
            
        i = len(nn.layers)
        string = string + "    Z" + str(i) + " = np.matmul(W" + str(i) + ", A" + str(i-1) + ") + b"+str(i)+"\n"
        string = string + "    A" + str(i) + " = self.softmax(Z" + str(i) + ")\n" #todo: review softmax 
        # Predictions        
        string = string + "    predictions = A" + str(i) + " > 0.5\n\n" #review this (wrong if softmax)
            
        string = string + "    return predictions\n"

        return string
    
        
    '''
    Creates and returns a dictionary with computed metrics (nescience, miscoding, surfeit, innacuracy,
    score and layer sizes), and enqueues it. Used in verbose mode. 
    '''
    def _update_vals(self, msdX):
        vals = dict()
        vals["nescience"]   = self._nescience(self.msd, self.viu, self.nn, msdX)
        vals["miscoding"]   = self._miscoding(self.msd, self.viu)
        vals["surfeit"]     = self._redundancy(self.nn)
        vals["inaccuracy"]  = self._inaccuracy(self.nn, msdX)
        vals["score"]       = self._score(self.nn, self.viu)
        vals["layer_sizes"] = [np.sum(self.viu)]
        vals["layer_sizes"] = vals["layer_sizes"] + self.nu
        # vals["layer_sizes"].append(1) 
        vals["layer_sizes"].append(self.n_classes) #fixed for multiclass 
        self.history.append(vals) #ofor google colab or notebooks (visualization)
        #queue.put(vals) #blocks sometimes
        # queue.put(vals["nescience"])
        return vals, queue

    
    """"
    Compute the length of a discrete variable given a minimal length code
    """
    def _codelength_discrete(self, data):
                
        unique, count = np.unique(data, return_counts=True)
        code  = np.zeros(self.n_classes)
        
        for i in np.arange(len(unique)):
            code[i] = - np.log2( count[i] / len(data) )

        #print(data)
        ldata = 0
        for i in np.arange(len(data)):
            ldata = ldata + code[data[i]]
            
        return ldata

    
    """"
    Compute the length of a continous variable given a minimal length code
    """    
    def _codelength_continuous(self, data):
        if len(np.unique(data)) == 1:
            Pred = np.zeros(len(data),dtype=np.int)
        else:
            nbins = int(np.sqrt(len(data)))
            tmp   = pd.qcut(data, q=nbins, duplicates='drop')
            Pred  = list(pd.Series(tmp).cat.codes)
            Pred = np.array(Pred)
                
        unique, count = np.unique(Pred, return_counts=True)
            
        code  = np.zeros(len(unique))
        
        for i in np.arange(len(unique)):
            code[i] = - np.log2( count[i] / len(Pred) )
        
        #temporal fix. Fails if tries ldata = np.sum(code[Pred])
        if(int(np.max(Pred)) == len(code)):
            code = np.append(code, 0)
        elif(int(np.max(Pred)) > len(code)):
            while(int(np.max(Pred)) >= len(code)): #this can be used (remove extra codes)
                code = np.append(code, 0)
        ldata = np.sum(code[Pred])

        # try: 
        #     ldata = np.sum(code[Pred]) 
        # except: 
        #     code = np.append(code,0) 
        #     print("EXCEPTION. Length of Pred: {}. Length of Code: {}.".format(len(Pred), len(code))) 
        #     ldata = np.sum(code[Pred])
        
        return ldata


    """
    Compute the conditional complexity of the target given a feature
          
    Return a numpy array with the conditional complexities
    """
    def _tcc(self):

            tcc = list()

            Resp = self.y
            unique, count_y = np.unique(Resp, return_counts=True)
            ldm_y = np.sum(count_y  * ( - np.log2(count_y  / len(Resp))))

            for i in np.arange(self.X.shape[1]):

                # Discretize the feature
                if len(np.unique(self.X[:,i])) == 1:
                    # Do not split if all the points belong to the same category
                    Pred = np.zeros(len(self.y))
                else:
                    nbins = int(np.sqrt(len(self.y)))
                    tmp   = pd.qcut(self.X[:,i], q=nbins, duplicates='drop')
                    Pred  = list(pd.Series(tmp).cat.codes)

                Join =  list(zip(Pred, Resp))

                unique, count_X  = np.unique(Pred, return_counts=True)
                unique, count_Xy = np.unique(Join, return_counts=True, axis=0)

                tot = self.X.shape[0]

                ldm_X   = np.sum(count_X  * ( - np.log2(count_X  / tot)))
                ldm_Xy  = np.sum(count_Xy * ( - np.log2(count_Xy / tot)))

                mscd = ( ldm_Xy - min(ldm_X, ldm_y) ) / max(ldm_X, ldm_y)

                tcc.append(mscd)

            return np.array(tcc)
    
    """
    Computes the confusion matrix for a given evaluation dataset to test the model.

    Prints Accuracy, Precision and Recall metrics.
    """
    def get_model_scores(self,X_test,y_test):
        if(len(np.unique(y_test)) > 2):
            print("Multiclass target. Not implemented yet Conf.Matrix.")
            scores = self.nn.evaluate(X_test[:,np.where(self.viu)[0]], to_categorical(y_test))
            print("Keras-Scores obtained: {}:{}, {}:{}.".format(self.nn.metrics_names[0],scores[0],self.nn.metrics_names[1],scores[1]))

            return
        y_pred = self.nn.predict(X_test[:,np.where(self.viu)[0]])
        y_test = to_categorical(y_test)
        matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

        true_n= matrix[0,0]
        false_n = matrix[1,0]
        false_p = matrix[0,1]
        true_p= matrix[1,1]

        print("True Positives:{}\nFalse Positives:{}\nFalse Negatives:{}\nTrue Negatives:{}".format(true_p,false_p,false_n,true_n))
        accuracy = (true_p + true_n) / (true_p + false_n+true_n + false_p)
        precision = (true_p) / (true_p + false_p)
        recall = (true_p) / (true_p + false_n )
        print("Accuracy: {}, Precision: {} , Recall: {}".format(accuracy,precision,recall))
        return

    """
    Plots Train and Test accuracy of the model obtained across train epochs.
    """
    def plot_model_acc(self):
        plt.plot(self.nn.history.history['acc'])
        plt.plot(self.nn.history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        return
    """
    Plots Train and Test loss of the model obtained across train epochs.
    """
    def plot_model_loss(self):
        plt.plot(self.nn.history.history['loss'])
        plt.plot(self.nn.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        return

    """
    Returns a dataframe containing algorithm history of models evaluated.

    Contains columns: 
    [h_l_sizes, #of nodes, nescience, inaccuracy, redundancy, Miscoding , Score]
    """
    def get_model_hist(self): 
        model_hist = pd.DataFrame(self.history)
        model_hist = model_hist.reset_index()
        model_hist.rename(columns={'index':'NN #'}, inplace=True)
        model_hist['h_l_nodes'] = model_hist['layer_sizes'].apply(lambda x: np.sum(x)-x[0]-self.n_classes) 
        return model_hist

    """
    Given a model dataframe history, plots Nescience and Inaccuracy, 
    Miscoding and Redundancy vs number of nodes used.
    """
    def vis_nescience(self,model_hist):
        fig, axes = plt.subplots(nrows=2, ncols=1)
        model_hist.plot(x='h_l_nodes',y=['inaccuracy','surfeit','miscoding'],ax=axes[0],title="Nescience Evolution through NAS")
        model_hist.plot(x='h_l_nodes', y=['nescience'],ax=axes[1])
        return