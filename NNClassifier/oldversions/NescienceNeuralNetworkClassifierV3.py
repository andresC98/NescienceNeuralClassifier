#@title
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Building optimal neural networks based on the minimum nescience principle

@author: Rafael Garcia Leiva
@mail:   rgarcialeiva@gmail.com
@web:    http://www.mathematicsunknown.com/
@copyright: All rights reserved
@version: 0.1 (13 Feb, 2018)

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
import collections
import time , sys
from random import randint

# Compression algorithms
import bz2
import lzma
import zlib

# Scikit-learn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier

# Keras 
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import Sequential
from keras import optimizers
from keras import losses
from keras.utils import to_categorical
from keras import backend as K

from queue import Queue

queue = Queue(10)
import os
import tensorflow as tf

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
        
        self.classes_  = None
        self.n_classes = None

        self.tol   = 0.05 #originally at 0.05
        self.decay = 0.1 #originally at 0.1

        self.output_num = None #TODO remove, we already have nclasses attribute 
        self.history = None #Records history of Stats for each algorithm' saved NN

    def fit(self, X, y):
        """
        Fit a model (a neural network with a hidden layer) given a dataset
    
        The input dataset has to be in the following format:
    
           * X = list([[x11, x12, x13, ...], ..., [xn1, xn2, ..., xnm]])
           * y = list([y1, ..., yn])
       
        Return the fitted model
        """
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
        # TODO: check the input parameters [DONE ?]
        if(len(X.shape) != 2 or len(y.shape) != 1):
            print("Invalid data shape/s of {} and output of {}.\nInput Array must be of format [[x11, x12, x13, ...], ..., [xn1, xn2, ..., xnm]].\nOutput array must be of format:[y1, ..., yn]".format(X.shape, y.shape))
            return
            #TODO Exit program?

        # TODO: test_size should be large enought, not a percentage of data
        self.X, self.X_test, self.y, self.y_test = train_test_split(X, y, test_size=0.3)

        self.X      = np.array(self.X)
        self.X_test = np.array(self.X_test)
        self.y      = np.array(self.y)
        self.y_test = np.array(self.y_test)

        self.classes_  = np.unique(y)
        self.n_classes = self.classes_.shape[0]

        self.history = [] #no need to be a synchr. queue as it wont be used for real-time graphs 

        #time metrics
        start_time = time.time()

        # Compute the contribution of each feature to miscoding
        
        self._initmiscod()
        
        if self.backward:
            self.viu = np.ones(self.X.shape[1], dtype=np.int)
        else:
            self.viu = np.zeros(self.X.shape[1], dtype=np.int)

        # Create the initial neural network
        #  - two features
        #  - one hidden layer
        #  - three units

        if self.backward:
            self.viu[np.where(self.msd == np.max(self.msd))] = 0
        else:            
            self.viu[np.where(self.msd == np.min(self.msd))] = 1

        msd = self.msd.copy()
        
        if self.backward:
            msd[np.where(self.viu == 0)] = 0
        else:
            msd[np.where(self.viu)] = 1

        if self.backward:
            self.viu[np.where(msd == np.max(msd))] = 0
        else:
            self.viu[np.where(msd == np.min(msd))] = 0

        msdX = self.X[:,np.where(self.viu)[0]]

        self.output_num = len(np.unique(self.y)) #this depends on dataset used, review.
        #Initial NN construction
        self.nn = Sequential()
        self.nn.add(Dense(units = self.nu[0], activation='relu', input_dim=msdX.shape[1]))
        self.nn.add(Dense(units = self.output_num, activation = 'softmax'))
        #adam = optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        sgd = optimizers.SGD(lr = self.lr, momentum = 0.9,nesterov = True)
        self.nn.compile(loss = losses.categorical_crossentropy ,optimizer = sgd, metrics=['accuracy'])
        #while keras being tested with multiclass classification - softmax
        self.y = to_categorical(self.y) #to use with categorical cross entropy
        self.y_test = to_categorical(self.y_test) #to use with categorical cross entropy
        
        #NN fit
        print("[DEBUG] Fitting initial NN.")
        self.nn.fit(x = msdX, y= self.y, verbose=0,batch_size = 32, epochs = self.it)
        
        self.nn.summary()
        print("[DEBUG] Evaluating initial nn...")
        score = self.nn.evaluate(msdX, self.y)
        print("[DEBUG] Keras-Scores obtained: {}:{}, {}:{}.".format(self.nn.metrics_names[0],score[0],self.nn.metrics_names[1],score[1]))
        self.nsc = self._nescience(self.msd, self.viu, self.nn, msdX)

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
        while (decreased):
            
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
                    msd[np.where(viu == 0)] = 0
                else:
                    msd[np.where(viu)] = 1

                if self.backward:
                    viu[np.where(msd == np.max(msd))] = 0
                else:
                    viu[np.where(msd == np.min(msd))] = 1

                msdX = self.X[:,np.where(viu)[0]]
                #New Keras NN creation
                print("[DEBUG] Testing adding a new feature to cnn. {} nº of features.".format(msdX.shape[1]))
                init_t = time.time()

                cnn = Sequential() #network has to be created from scratch, we cannot reuse weights
                cnn.add(Dense(units = self.nu[0], activation='relu', input_dim=msdX.shape[1])) #first layer after inputs
                for layer in np.arange(len(self.nu)-1):
                    cnn.add(Dense(units = self.nu[layer+1], activation = 'relu'))
                cnn.add(Dense(units = self.output_num, activation = 'softmax'))
                # adam = optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
                sgd = optimizers.SGD(lr = self.lr, momentum = 0.9, nesterov = True)
                cnn.compile(loss = losses.categorical_crossentropy ,optimizer = sgd, metrics=['accuracy'])
                cnn.fit(x = msdX, y= self.y, verbose=0,batch_size = 32, epochs = self.it)

                network_comp_t = time.time() - init_t
                print("Took {:.2f}s. to create and fit test NN.".format(network_comp_t))
                
                nsc = self._nescience(self.msd, viu, cnn, msdX)

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
                    cnn.summary()

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
            cnn = Sequential()
            cnn.add(Dense(units = nu[0], activation='relu', input_dim=msdX.shape[1]))
            for k, units in enumerate(nu):
                if(k > 0): #we already added the first hidden layer
                    cnn.add(Dense(units, activation = 'relu'))
            
            cnn.add(Dense(units = self.output_num, activation = 'softmax'))
            #reusing weights from previous saved network's layers.
            # cnn = Sequential()
            # for layer in self.nn.layers[:-1]: # just exclude last layer from copying
            #     cnn.add(layer)
            # for layer in cnn.layers:
            #     layer.trainable = False #to reuse their weights

            # cnn.add(Dense(units = nu[-1], activation = 'relu')) #adding the extra layer
            # cnn.add(Dense(units = self.output_num, activation = 'softmax'))
            sgd = optimizers.SGD(lr = self.lr, momentum = 0.9,nesterov = True)
            # adam = optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.1, amsgrad=False)
            cnn.compile(loss = losses.categorical_crossentropy ,optimizer = sgd, metrics=['accuracy'])
            cnn.fit(x = msdX, y= self.y, verbose=0,batch_size = 32, epochs = self.it)
            
            network_comp_t = time.time() - init_t
            print("Took {:.2f}s. to create and fit test NN.".format(network_comp_t))

            nsc = self._nescience(self.msd, self.viu, cnn, msdX)


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
                cnn.summary()
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

            #
            # Test adding a new unit
            # for each layer: adds a unit, check nescience
            
            #replicates previous network:
            # for i in np.arange(len(self.nu)):
            #     nu = self.nu.copy()
            #     nu[i] = nu[i] + 1 #extra node added to hidden layer i     
            #     print("[DEBUG] Testing adding a new unit... Hidden layers: {}".format(nu))            
            #     msdX = self.X[:,np.where(self.viu)[0]]
                
            #     for k in np.arange(i): #iterate through layers
            #         cnn = Sequential()
            #         for layer_no, layer in enumerate(self.nn.layers[:-1]): # just exclude last layer from copying
            #             if(layer_no > k): #we can reuse previous layers
            #                 cnn.add(layer)
            #             else: 
            #                 break

            #         for layer in cnn.layers:
            #             layer.trainable = False #to reuse their weights
            #                 cnn.add(Dense(units = self.output_num, activation = 'softmax'))

            #          cnn.add(Dense(units = self.output_num, activation = 'softmax'))



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
                cnn.add(Dense(units = self.output_num, activation = 'softmax'))
                sgd = optimizers.SGD(lr = self.lr, momentum = 0.9, nesterov = True)
                # adam = optimizers.Adam(lr= self.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)   
                cnn.compile(loss = losses.categorical_crossentropy ,optimizer = sgd, metrics=['accuracy'])
                cnn.fit(x = msdX, y= self.y, verbose=0,batch_size = 32, epochs = self.it)

                network_comp_t = time.time() - init_t
                print("Took {:.2f}s. to create and fit test NN.".format(network_comp_t))

                nsc = self._nescience(self.msd, self.viu, cnn, msdX)

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
                    cnn.summary()
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

        print("Proceeding to test obtained network with whole dataset:")
        final_score = self.score(X[:,np.where(self.viu)[0]],to_categorical(y))
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
        print("Neural Network to test:")
        self.nn.summary()
        score = self.nn.evaluate(X, y, verbose=1)[1] #evaluate returns (loss, accuracy) tuple
        print("Obtained Network Evaluation Score of: {}".format(score))
        
        return score      

    
    """
    Compute the contribution of each feature to miscoding
    
    TODO: Somehow we should penalize non-contributing features
    """
    def _initmiscod(self):
         
        msd = list()

        for i in range(0, self.X.shape[1]):

            Pred = list(pd.cut(self.X[:,i], bins=100, labels=range(0, 100)))
            Resp = self.y
            Join =  list(zip(Pred, Resp))

            count_X  = collections.Counter(Pred)
            count_y  = collections.Counter(Resp)
            count_Xy = collections.Counter(Join)
    
            tot_X = self.X.shape[0]

            ldm_X  = 0
            for key in count_X.keys():
                ldm_X = ldm_X + count_X[key] * ( - np.log2(count_X[key] / tot_X))
    
            ldm_y  = 0
            for key in count_y.keys():
                ldm_y = ldm_y + count_y[key] * ( - np.log2(count_y[key] / len(self.y)))

            ldm_Xy  = 0
            for key in count_Xy.keys():
                ldm_Xy = ldm_Xy + count_Xy[key] * ( - np.log2(count_Xy[key] / len(self.y)))
       
            K_yX = ldm_Xy - ldm_X
            # K_Xy = ldm_Xy - ldm_y
    
            # TODO: It should be something like:
            # mscd = max(K_yX, K_Xy) / max(ldm_X, ldm_y) / self.X.shape[1]
            
            mscd = K_yX / ldm_y 

            msd.append(mscd)

        # self.msd = np.array(msd) / np.sum(msd)
        self.msd = np.array(msd)
        return    

    
    """
    Compute the miscoding of the dataset used by the current model
      
    Return the miscoding
    """
    def _miscoding(self, msd, viu):
        
        mean = np.mean(msd)
        mmsd = mean - msd
        mmmsd = np.zeros(len(msd))
        
        mmmsd[np.where(mmsd > 0)] = mmsd[np.where(mmsd > 0)] / np.sum(mmsd[np.where(mmsd > 0)])
        mmmsd[np.where(mmsd < 0)]  = mmsd[np.where(mmsd < 0)] / np.abs(np.sum(mmsd[np.where(mmsd < 0)]))

        # miscoding = np.abs(1 - np.sum(np.multiply(1/len(self.viu) - msd, viu)))
        # miscoding = np.sum(np.multiply(msd, viu))
        miscoding = 1 - np.sum(np.multiply(mmmsd, viu))
                    
        return miscoding
 
    
    """
    Compute inaccuracy the current tree

    Warning: If the inaccuracy is a negative number it should not be used

    TODO: Perhaps I should add y[i] to the error string
      
    Return the inaccuracy
    """
    def _inaccuracy(self, nn, X):
                        
        # Compute the list of errors
        
        pred = self.predict(nn, X)
        
        error = list()
        #print("[DEBUG] Y shape: {}, Pred shape: {}".format(self.y.shape, pred.shape))
        for i in range(X.shape[0]):
            if pred[i] != np.argmax(self.y, axis=1)[i]:
                error.append(list(X[i,:]))

        # Compute the length of the encoding of the error
        error  = str(error).encode()
        
        if self.compressor == "lzma":
            dmodel = lzma.compress(error, preset=9)
        elif self.compressor == "zlib": 
            dmodel = zlib.compress(error, level=9)
        else: # By default use bz2   
            dmodel = bz2.compress(error, compresslevel=9)        
        
        ldm    = len(dmodel)
        
        # Check if the error is too small to compress
        if ldm >= len(error):
            return self.INVALID_INACCURACY

        # Compute the length of the encoding of the dataset
        data  = (str(X.tolist()) + str(self.y.tolist())).encode()
        data  = bz2.compress(data, compresslevel=9)
        ld    = len(data)
                
        # Inaccuracy = l(d/m) / l(d)
        inaccuracy = ldm / ld
        
        return inaccuracy

    """
    Compute the redundancy of the current tree
    
    Warning: If the redundancy is a negative number it should not be used
         
    Return the redundancy of the tree
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
    Compute the nescience of a tree,
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

        x_test = self.X_test[:,np.where(viu)[0]]
        score = self.nn.evaluate(x_test, self.y_test)[1]
        print("[DEBUG] NN Evaluated Accuracy = {}. Precision: {}. Recall: {}".format(score))
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
    
            
    def _display_nn(self, nn, hidden_units):

        self.canvas_size = 400
        self.top    = .9 * self.canvas_size
        self.bottom = .1 * self.canvas_size
        self.left   = .1 * self.canvas_size
        self.right  = .9 * self.canvas_size

        self.canvas.delete(ALL)
        
        layer_sizes = [np.sum(self.viu)]
        layer_sizes = layer_sizes + hidden_units
        # layer_sizes.append(1) #fixed for multiclass classification (softmax)
        layer_sizes.append(self.output_num)
        v_spacing = (self.top - self.bottom)/float(max(layer_sizes))
        h_spacing = (self.right - self.left)/float(len(layer_sizes) - 1)
        
        # Edges
    
        for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        
            layer_top_a = v_spacing * (layer_size_a - 1)/2. + (self.top + self.bottom)/2.
            layer_top_b = v_spacing * (layer_size_b - 1)/2. + (self.top + self.bottom)/2.
                        
            for m in np.arange(layer_size_a):
                for o in np.arange(layer_size_b):
                    self.canvas.create_line(n * h_spacing + self.left,
                                            layer_top_a - m * v_spacing,
                                            (n + 1) * h_spacing + self.left,
                                            layer_top_b - o * v_spacing)

        # Nodes

        for n, layer_size in enumerate(layer_sizes):
    
            layer_top = v_spacing * (layer_size - 1)/2. + (self.top + self.bottom)/2.
        
            for m in np.arange(layer_size):
                
                self.canvas.create_oval(n * h_spacing + self.left - v_spacing/4,
                                        layer_top - m * v_spacing - v_spacing/4,
                                        n * h_spacing + self.left + v_spacing/4,
                                        layer_top - m * v_spacing + v_spacing/4,
                                        fill="white")
                
        self.master.update_idletasks()
        self.master.update()

    '''
    Creates and returns a dictionary with computed metrics (nescience, miscoding, surfeit, innacuracy,
    score and layer sizes), and enqueues it. Used in verbose mode. 
    '''
    def _update_vals(self, msdX):
        # init_t = time.time()
        vals = dict()
        vals["nescience"]   = self._nescience(self.msd, self.viu, self.nn, msdX)
        vals["miscoding"]   = self._miscoding(self.msd, self.viu)
        vals["surfeit"]     = self._redundancy(self.nn)
        vals["inaccuracy"]  = self._inaccuracy(self.nn, msdX)
        vals["score"]       = self._score(self.nn, self.viu)
        vals["layer_sizes"] = [np.sum(self.viu)]
        vals["layer_sizes"] = vals["layer_sizes"] + self.nu
        # vals["layer_sizes"].append(1) 
        vals["layer_sizes"].append(self.output_num) #fixed for multiclass 
        #self.history.append(vals) #only for google colab or notebooks
        queue.put(vals) #blocks sometimes
        # queue.put(vals["nescience"])
        # elapsed_t = init_t - time.time()
        # print("[DEBUG] It took {:.2f}s. to evaluate test NN.".format(elapsed_t))
        return vals, queue

    
    """"
    Compute the length of a discrete variable given a minimal length code
    """
    def _codelength_discrete(self, data):
                
        unique, count = np.unique(data, return_counts=True)
        code  = np.zeros(self.n_classes)
        
        for i in np.arange(len(unique)):
            code[i] = - np.log2( count[i] / len(data) )

        ldata = 0
        for i in np.arange(len(data)):
            ldata = ldata + code[data[i]]
            
        return ldata

    
    """"
    Compute the length of a continous variable given a minimal length code
    """    
    def _codelength_continuous(self, data):

        if len(np.unique(data)) == 1:
            Pred = np.zeros(len(data))
        else:                
            nbins = int(np.sqrt(len(data)))
            tmp   = pd.qcut(data, q=nbins, duplicates='drop')
            Pred  = list(pd.Series(tmp).cat.codes)
                
        unique, count = np.unique(Pred, return_counts=True)
        code  = np.zeros(len(unique))
        
        for i in np.arange(len(unique)):
            code[i] = - np.log2( count[i] / len(Pred) )

        ldata = np.sum(code[Pred])            
            
        return ldata
