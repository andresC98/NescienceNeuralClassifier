"""

Compute the nescience of a entity given an encoding dataset and a model

@author:    Rafael Garcia Leiva
@mail:      rgarcialeiva@gmail.com
@web:       http://www.mathematicsunknown.com/
@copyright: GNU GPLv3
@version:   0.1 (Jun 2019)

TODO:
    - Program fails if X has only one attribute
    
    Other tasks to do include:
    - Miscoding should be computed based on groups of features        
    - Provide support to regression problems
    - Allow categorical features
    - Allow missing data
    - Work in parallel

"""

import numpy  as np
import pandas as pd
import math

import bz2
import lzma
import zlib

class Nescience:
    
    def __init__(self, X, y, verbose=False, method="Harmonic", compressor="zlib"):
        """
        Initialization of the class nescience
        
        The input dataset has to be in the following format:
    
           * X = list([[x11, x12, x13, ...], ..., [xn1, xn2, ..., xnm]])
           * y = list([y1, ..., yn])

        Other optional parameters are:

          * verbose (boolean):   If true, prints out additional information
          * method (string):     method used to comput the nescience. Valid
                                 values are: "Euclid", "Arithmetic",
                                 "Geometric", "Product", "Addition" and
                                 "Harmonic".
          * compressor (string): compressor used to compute redudancy. Valid
                                 values are: "bz2", "lzma" and "zlib".
          
        """

        # TODO: check the input parameters and init arguments
        
        self.verbose      = verbose
        self.method       = method
        self.compressor   = compressor

        self.X = np.array(X)
        self.classes_, self.y = np.unique(y, return_inverse=True)
        self.n_classes = self.classes_.shape[0]
        
        # Compute the optimal code lengths for the attributes and target
        
        if verbose == True:
            print("Computing optimal codes ... ", end="")
        
        self.lcdY = self._codelength_discrete(self.y)
        
        self.lcdX = np.zeros(self.X.shape[1])
        for i in np.arange(self.X.shape[1]):                
            self.lcdX[i] = self._codelength_continuous(self.X[:,i])
        
        if verbose == True:
            print("done!")
                
        # Compute the contribution of each feature to miscoding         
        
        if verbose == True:
            print("Computing miscoding ... ", end="")
        
        self.tcc = self._tcc()
        
        # Used internally to compute the model miscoding
        
        self.norm_mscd = 1 - np.array(self.tcc)
        self.norm_mscd = self.norm_mscd / np.sum(self.norm_mscd)

        if verbose == True:
            print("done!")
                    
        return None

    
    """
    Compute the global miscoding of the dataset used by the current tree
      
    Return the miscoding (float)
    """
    def miscoding(self, att_in_use):

        miscoding = np.dot(att_in_use, self.norm_mscd)
        miscoding = 1 - miscoding
                            
        return miscoding


    def inaccuracy(self, errors, attr_in_use):
        """
        Compute the inaccuracy of the model
        
         * errors - list of errors
         
        Return the inaccuracy (float)
        """
        
        ldm   = 0
        ldata = 0

        # Compute the compressed length of errors
            
        for i in np.arange(len(errors[0])-1):
            ldm = ldm + self._codelength_continuous(errors[:,i])
        
        ldm = ldm + self._codelength_discrete(errors[:,-1].astype(np.int))
        
        # Compute the compressed length of data in use
            
        for i in np.arange(self.X.shape[1]):

            if attr_in_use[i] == 0:
                continue

            ldata = ldata + self.lcdX[i]
        
        ldata = ldata + self.lcdY
            
        inaccuracy = ldm / ldata
            
        return inaccuracy


    def redundancy(self, model):
        """
        Compute the redundancy of a model
        
         * model -  a string based represenation of a model
    
        Return the redundancy (float)
        """
    
        # Compute the model string and its compressed version
        emodel = model.encode()
        
        if self.compressor == "lzma":
            compressed = lzma.compress(emodel, preset=9)
        elif self.compressor == "zlib":
            compressed = zlib.compress(emodel, level=9)
        else: # By default use bz2
            compressed = bz2.compress(emodel, compresslevel=9)
        
        km = len(compressed)
        lm = len(emodel)
        
        if km > lm:
            # Model cannot be compressed
            km = lm
        else:
            # redundancy = 1 - l(m*) / l(m)
            redundancy = 1 - km / lm
            
        return redundancy


    """
    Compute the nescience of a model and a dataset
              
    Return the nescience (float)
    """
    def nescience(self, attr_in_use, errors, model):

        miscoding  = self.miscoding(attr_in_use)
        inaccuracy = self.inaccuracy(errors, attr_in_use)
        redundancy = self.redundancy(model)

        if redundancy == 0:
            # Avoid dividing by zero
            redundancy = 10e-6

        # TODO: Provide a theoretical interpretation of this decision
        if redundancy < inaccuracy:
            # The model is still too small to compute the nescience
            # use innacuracy instead
            redundancy = 1
    
        if inaccuracy == 0:
            # Avoid dividing by zero
            inaccuracy = 10e-6

        if miscoding == 0:
            # Avoid dividing by zero
            miscoding = 10e-6

        # Compute the nescience according to the method specified by the user
        if self.method == "Euclid":
            # Euclidean distance
            nescience = math.sqrt(miscoding**2 + inaccuracy**2 + redundancy**2)
        elif self.method == "Arithmetic":
            # Arithmetic mean
            nescience = (miscoding + inaccuracy + redundancy) / 3
        elif self.method == "Geometric":
            # Geometric mean
            nescience = math.pow(miscoding * inaccuracy * redundancy, 1/3)
        elif self.method == "Product":
            # The product of both quantities
            nescience = miscoding * inaccuracy * redundancy
        elif self.method == "Addition":
            # The product of both quantities
            nescience = miscoding + inaccuracy + redundancy
        elif self.method == "Weighted":
            # Weigthed sum
            nescience = self.weight_miscoding * miscoding + self.weight_inaccuracy * inaccuracy + self.weight_surfeit * redundancy
        else:
            # By default use the Harmonic mean
            if inaccuracy == 0:
                # Avoid dividing by zero
                inaccuracy = np.finfo(np.float32).tiny
            nescience = 3 / ( (1/miscoding) + (1/inaccuracy) + (1/redundancy))
            
        return nescience


    def targetconditionalcomplexity(self):
        """
        Return the normalized conditional complexity of the target
        given the features
          
        Return a numpy array with the normalized conditional complexities
        """
        
        return self.norm_mscd


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
