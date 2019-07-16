#Genetic Algorithm Library (PonyGE2)
from fitness.base_ff_classes.base_ff import base_ff
import stats
#Math, Data Science and System libraries
import numpy  as np
import pandas as pd
import math, collections
import time , sys, os, threading
from random import randint
import matplotlib.pyplot as plt

# Compression algorithms
import bz2, lzma, zlib
# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
# Keras & tensorflow
import keras
from keras.layers import Dense 
from keras.models import Model, Sequential
from keras import optimizers, losses
from keras.utils import to_categorical
from keras import backend as K

import tensorflow as tf

from queue import Queue
queue = Queue(10)

import logging, csv, traceback
from datetime import datetime

class NescNNclasGE(base_ff):
    maximise = False #we want to minimize our objective: Nescience value.
    global queue
    # Constants defintion
    INVALID_INACCURACY = -1    # If algorithm cannot be applied to this dataset
    INVALID_REDUNDANCY = -2    # Too small model    
    
    def __init__(self):

        #Disabling (annoying) TF Info logs
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.get_logger().setLevel(logging.ERROR)
        tf.logging.set_verbosity(tf.logging.ERROR)
        super().__init__()

        #class (fixed) attributes
        self.it         = 25
        self.lr         = 0.01
        self.method     = "Harmonic"
        self.compressor = "bz2"
        self.backward   = False
        self.verbose    = True
        self.optimizer  = None
    
        self.nsc_csv_name = datetime.now().strftime('%H-%M-%S.csv')

        #Data preprocessing
        scaler = StandardScaler()

        data = load_digits()
        X = data.data
        y = data.target
        
        self.classes_, self.y = np.unique(y, return_inverse=True)
        self.n_classes = self.classes_.shape[0]
        self.X, self.X_test, self.y, self.y_test  = train_test_split(X,self.y, test_size = 0.2)

        self.X = scaler.fit_transform(self.X)
        self.X_test = scaler.transform(self.X_test)
        self.X = np.array(self.X)
        self.X_test = np.array(self.X_test)

        self.n_vars = self.X.shape[1]
        #Initial miscoding computation

        self.lcdY = self._codelength_discrete(self.y)
        self.lcdX = np.zeros(self.X.shape[1])
        for i in np.arange(self.X.shape[1]):
            self.lcdX[i] = self._codelength_continuous(self.X[:,i])

        self.lcdY_test = self._codelength_discrete(self.y_test)
        self.lcdX_test = np.zeros(self.X_test.shape[1])
        for i in np.arange(self.X_test.shape[1]):
            self.lcdX_test[i] = self._codelength_continuous(self.X_test[:,i])

        self.tcc = self._tcc()
        norm_mscd = 1 - np.array(self.tcc)
        self.norm_mscd = norm_mscd / np.sum(norm_mscd)
        msd = self.norm_mscd.copy() 

        print("Computing enhanced miscoding of dataset...")
        self.miscoding, self.vius = self.compute_mscd_list()

        self.nsc  = None #Nescience of current individual
        self.y = to_categorical(self.y)
        self.y_test = to_categorical(self.y_test)  
        self.nn = None #Model (that will be tested, etc)

    def compute_mscd_list(self):
        '''
        Computes miscoding list and the corresponding list of vius.
        '''
        #Variables in use
        viu = np.zeros(self.X.shape[1],dtype=np.int)
        #List of viu arrays 
        vius = []
        msd = self.norm_mscd.copy()
        msd_list = []

        for i  in np.arange(self.X.shape[1]):
            msd[np.where(viu)] = 0
            viu[np.where(msd == np.max(msd))] = 1
            vius.append(viu.copy())
            enhanced_mscd = self._enhanced_miscoding(viu)
            msd_list.append(enhanced_mscd)
        

        #Array containing miscoding from 1 to N attributes in use
        miscoding = np.array(msd_list)

        return miscoding, vius


        return mscd 

    def evaluate(self, ind, **kwargs):
        inargs = {"self": self}
        try:
            exec(ind.phenotype,inargs) #msdX, self.nn and opt. initialized here
            #Obtain generated output from exec dictionary
            self.viu = inargs['viu']
            self.nn = inargs['nn']
            msdX = inargs['msdX']
            self.msdX = msdX
            self.optimizer = self._create_optimizer(inargs['opt'])

            #Once GE has decided model, proceed to compile, test and evaluate it.
            self.nn.compile(loss = losses.categorical_crossentropy ,optimizer = self.optimizer, metrics=['accuracy'])
            self.nn.fit(x = msdX, y= self.y, validation_split=0.33,verbose=0,batch_size = 32, epochs = self.it)

            #Nescience computations should be done using Test data.
            msdX_test = self.X_test[:,np.where(self.viu)[0]]
            nsc = self._nescience(self.viu, self.nn, msdX_test)
            vals = self._update_vals(msdX_test)
            
            nsc_data = [ind.name, vals["inaccuracy"],vals["surfeit"], vals["nescience"]]
            with open("./nsc_results/"+self.nsc_csv_name, 'a') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(nsc_data)
            num_vius = np.count_nonzero(self.viu == 1)

            print("[",num_vius,"VIUs] Inaccuracy: ", vals["inaccuracy"],"Score:",vals["score"],"Miscoding:",vals["miscoding"],"Redundancy:", vals["surfeit"], "Nescience:", vals["nescience"])

                
        except:
            traceback.print_exc()
            nsc = 0.99 #invalid network has high nsc (very bad)

        return nsc #this will be the target to minimize by the GE algorithm.

    def _create_optimizer(self, opt):
        if "sgd" in opt:
            return optimizers.SGD(lr = self.lr, momentum = 0.9,nesterov = True)
        if "adam" in opt:
            return optimizers.adam()

    def _nescience(self, viu, nn, X):
        
        miscoding  = self._enhanced_miscoding(viu)

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
            nescience = 2 / ( (1/inaccuracy) + (1/redundancy))
            #nescience = 3 / ( (1/miscoding) + (1/inaccuracy) + (1/redundancy))            

        return nescience
    
    def _enhanced_miscoding(self, att_in_use):
        tcc  = self.norm_mscd
        ctcc = self.tcc
        ctcc = ctcc / np.sum(ctcc)
        diff = tcc - ctcc

        miscoding = np.dot(att_in_use, diff)
        miscoding = 1 - miscoding

        return miscoding


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

    def _inaccuracy(self, nn, X):
                        
        # Compute the list of errors
        ldm = 0
        ldata = 0

        pred = self.predict(nn, X)

        errors = list()
        for i in np.arange(X.shape[0]):
            if(pred[i] != np.argmax(self.y_test, axis=1)[i]):
                new_error = list(X[i])
                new_error.append(np.argmax(self.y_test, axis=1)[i])
                errors.append(new_error)
        
        errors = np.array(errors)
        
        # Compute the compressed length of errors    
        for i in np.arange(len(errors[0])-1):
            ldm = ldm + self._codelength_continuous(errors[:,i])
        
        ldm = ldm + self._codelength_discrete(errors[:,-1].astype(np.int))
        # Compute the compressed length of data in use
            
        
        for i in np.arange(self.X_test.shape[1]):
            if self.viu[i] == 0:
                continue
            ldata = ldata + self.lcdX_test[i]
        
        ldata = ldata + self.lcdY_test
            
        inaccuracy = ldm / ldata
        
        return inaccuracy

    def _update_vals(self, msdX):

        vals = dict()
        vals["nescience"]   = self._nescience(self.viu, self.nn, msdX)
        vals["miscoding"]   = self._enhanced_miscoding(self.viu)
        vals["surfeit"]     = self._redundancy(self.nn)
        vals["inaccuracy"]  = self._inaccuracy(self.nn, msdX)
        vals["score"]       = self._score(self.nn, self.viu)
        vals["layer_sizes"] = [np.sum(self.viu)]
        vals["layer_sizes"] = vals["layer_sizes"]
        vals["layer_sizes"].append(self.n_classes) #fixed for multiclass 

        return vals

    def predict(self, nn, X):
        """
        Predict class given a dataset
    
          * X = list([[x11, x12, x13, ...], ..., [xn1, xn2, ..., xnm]])
    
        Return a list of classes predicted
        """
        # TODO: Check that we have a model trained
        predictions = nn.predict(X) # Softmax output
        predictions = np.argmax(predictions, axis=1)
        return predictions

    def _score(self, nn, viu):

        x = self.X_test[:,np.where(viu)[0]]
        score = self.nn.evaluate(x, self.y_test, verbose=0)
        return score[1]

    def _nn2str(self, nn):
        # TODO: Review 
        # 10/06/19: Solved incorrect index printing of A_i, Z_i, W_i
        # 12/06/19: Migrated nn2str code for Keras model compatibility
        # Header
        string = "def NN(X):\n"
            
        for i in np.arange(len(nn.layers)):
            if(not nn.layers[i].get_weights()): #dropout layer. Skip.
                continue #dropout "layer" has no weights.
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

        return ldata

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