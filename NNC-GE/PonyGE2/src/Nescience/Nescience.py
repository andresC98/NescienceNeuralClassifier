"""

Compute the nescience of a entity given an encoding dataset and a model

@author:    Rafael Garcia Leiva
@mail:      rgarcialeiva@gmail.com
@web:       http://www.mathematicsunknown.com/
@copyright: GNU GPLv3
@version:   0.1

"""

import numpy  as np
import pandas as pd

import math

from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin

from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted

# Compressors

import bz2
import lzma
import zlib

# Supported models

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

#
# Helper Functions
#

def discretize(x):
    """
    Discretize the variable x if needed
    
    Parameters
    ----------
    x : array-like, shape (n_samples)
        The variable to be discretized, if needed.
       
    Returns
    -------
    Return the miscoding (float)
    """
        
    if len(np.unique(x)) > int(np.sqrt(len(x))):
        # Too many unique values wrt samples
        nbins = int(np.sqrt(len(x)))
        tmp   = pd.qcut(x, q=nbins, duplicates='drop')
        new_x = list(pd.Series(tmp).cat.codes)
    else:
        new_x = x
        
    return new_x
    

def optimal_code_length(x):
    """
    Compute the lenght of a list of values encoded using an optimal code
    
    Parameters
    ----------
    x : array-like, shape (n_samples)
        The values to be encoded.
       
    Returns
    -------
    Return the length of the encoded dataset (float)
    """
    
    # Discretize the variable x if needed
    
    new_x = discretize(x)
        
    # Compute the optimal length
        
    unique, count = np.unique(new_x, return_counts=True)
    ldm = np.sum(count * ( - np.log2(count / len(new_x))))
                    
    return ldm


def optimal_code_length_join(x1, x2):
    """
    Compute the lenght of the join of two variable
    encoded using an optimal code
    
    Parameters
    ----------
    x1 : array-like, shape (n_samples)
         The values of the first variable.
         
    x2 : array-like, shape (n_samples)
         The values of the second variable.         
       
    Returns
    -------
    Return the length of the encoded join dataset (float)    
    """
    
    # Discretize the variables X1 and X2 if needed
        
    new_x1 = discretize(x1)
    new_x2 = discretize(x2)
    
    # Compute the optimal length
    Join =  list(zip(new_x1, new_x2))
    unique, count = np.unique(Join, return_counts=True, axis=0)
    ldm = np.sum(count * ( - np.log2(count / len(Join))))
                    
    return ldm

#
# Class Miscoding
# 

class Miscoding(BaseEstimator, SelectorMixin):
    
    # TODO: Class documentation
    
    def __init__(self):
        
        return None
    
    
    def fit(self, X, y):
        """
        Learn empirically the miscoding of the features of X
        as a representation of y.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Sample vectors from which to compute miscoding.
            
        y : array-like, shape (n_samples)
            The target values (class labels) as numbers or strings.
            
        Returns
        -------
        self
        """
        
        self.X, self.y = check_X_y(X, y, dtype=None)

        # Regular miscoding
        self.regular = self._regular_miscoding()

        # Adjusted miscoding
        self.adjusted = 1 - self.regular
        self.adjusted = self.adjusted / np.sum(self.adjusted)

        # Partial miscoding
        self.partial = self.adjusted - self.regular / np.sum(self.regular)

        return self

    
    # TODO: do we have to support this?
    def _get_support_mask(self):
        
        check_is_fitted(self, 'tcc_')
        
        return None


    def miscoding_features(self, type='adjusted'):
        """
        Return the miscoding of the target given individual features

        Parameters
        ----------
        type : the type of miscoding we want to compute, possible values
               are 'regular', 'adjusted' and 'partial'.
            
        Returns
        -------
        Return a numpy array with the miscodings
        """
        
        check_is_fitted(self, 'regular')
        
        if type == 'regular':
            return self.regular
        elif type == 'adjusted':
            return self.adjusted
        elif type == 'partial':
            return self.partial
        
        # TODO: rise exception
        return None


    def miscoding_model(self, model):
        """
        Compute the global miscoding of the dataset given a model
        
        Parameters
        ----------
        model : a model of one of the supported classeses
        
        Supported classes:
            DecisionTreeClassifier
            
        Returns
        -------
        Return the miscoding (float)
        """
        
        if isinstance(model, DecisionTreeClassifier):
            subset = self._DecisionTreeClassifier(model)
        else:
            # TODO: Rise exception
            return None

        return self.miscoding_subset(subset)
        

    def miscoding_subset(self, subset):
        """
        Compute the global miscoding of a subset of the dataset
        
        Parameters
        ----------
        subset : array-like, shape (n_features)
                 1 if the attribute is in use, 0 otherwise
            
        Returns
        -------
        Return the miscoding (float)
        """
        
        # TODO: check the format of subset
        
        miscoding = np.dot(subset, self.partial)
        miscoding = 1 - miscoding
                       
        return miscoding

    
    """    
    Compute the regular miscoding of each feature
          
    Return a numpy array with the regular miscodings
    """
    def _regular_miscoding(self):
         
        miscoding = list()
        
        # Discretize y and compute the encoded length
        
        Resp  = discretize(self.y)
        ldm_y = optimal_code_length(self.y)

        for i in np.arange(self.X.shape[1]):

            # Discretize feature and compute lengths
            
            Pred  = discretize(self.X[:,i])
            ldm_X = optimal_code_length(self.X[:,i])
            
            ldm_Xy = optimal_code_length_join(Resp, Pred)

            # Compute miscoding
                       
            mscd = ( ldm_Xy - min(ldm_X, ldm_y) ) / max(ldm_X, ldm_y)
                
            miscoding.append(mscd)
                
        return np.array(miscoding)


    """
    Compute the attributes in use for a decision tree
    
    Return array with the attributes in use
    """
    def _DecisionTreeClassifier(self, estimator):

        # TODO: sanity check over estimator

        attr_in_use = np.zeros(self.X.shape[1], dtype=int)
        features = set(estimator.tree_.feature[estimator.tree_.feature >= 0])
        for i in features:
            attr_in_use[i] = 1
            
        return attr_in_use

#
# Class Inaccuracy
#

class Inaccuracy(BaseEstimator, SelectorMixin):
    
    # TODO: Class documentation
    
    def __init__(self):
        
        return None
    
    
    def fit(self, X, y):
        """Initialize the inaccuracy class with dataset
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Sample vectors from which models have been trained.
            
        y : array-like, shape (n_samples)
            The target values (class labels) as integers or strings.
            
        Returns
        -------
        self
        """
        
        self.X, self.y = check_X_y(X, y, dtype=None)
                
        self.len_y = optimal_code_length(self.y)
        
        return self


    def inaccuracy_model(self, model):
        """
        Compute the inaccuracy of a model
        
        model : trained model with a predict() method
         
        Return the inaccuracy (float)
        """        
        
        # TODO: sanity check of the model
        
        check_is_fitted(self, 'X')
  
        Pred = model.predict(self.X)
        len_pred = optimal_code_length(Pred)
        
        len_join = optimal_code_length_join(Pred, self.y)
        
        inacc = ( len_join - min(self.len_y, len_pred) ) / max(self.len_y, len_pred)

        return inacc

    
    def inaccuracy_predictions(self, predictions):
        """
        Compute the inaccuracy of a list of predicted values
        
         pred : array-like, shape (n_samples)
                The list of predicted values
                
        Return the inaccuracy (float)
        """        
        check_is_fitted(self, 'X')
        len_pred = optimal_code_length(predictions)
        
        len_join = optimal_code_length_join(predictions, self.y)
        
        inacc = ( len_join - min(self.len_y, len_pred) ) / max(self.len_y, len_pred)

        return inacc    


    # TODO
    def _get_support_mask(self):
        
        check_is_fitted(self, 'tcc_')
 
#
# Class Surfeit
# 
    
class Surfeit(BaseEstimator, SelectorMixin):
    
    def __init__(self):
        
        return None
    
    
    def fit(self, compressor="bz2"):
        """Learn empirically the surfeit of a model.
        
        Parameters
        ----------
        model : a trained model
            
        Returns
        -------
        self
        """
                
        self.compressor = compressor

        return self

    def surfeit_model(self, model):
        """
        Compute the redundancy of a model

        Parameters
        ----------
        model : a model of one of the supported classeses
        
        Supported classes:
            DecisionTreeClassifier
            MLPClassifier
            
        Returns
        -------
        Redundancy (float) of the model
        """
    
        if isinstance(model, DecisionTreeClassifier):
            model_str = self._DecisionTreeClassifier(model)
        elif isinstance(model, MLPClassifier):
            model_str = self._MLPClassifier(model)
        else: #keras case, but this is a placeholder
            # TODO: Rise exception
            model_str = self._Keras_MLPClassifier(model)
            #return None

        return self.surfeit_string(model_str)
        

    def surfeit_string(self, model_string):
        """
        Compute the redundancy of a model given as a string

        Parameters
        ----------
        model : a string based representation of the model
            
        Returns
        -------
        Redundancy (float) of the model
        """
    
        # Compute the model string and its compressed version
        emodel = model_string.encode()
        
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
        
        # redundancy = 1 - l(m*) / l(m)
        redundancy = 1 - km / lm
            
        return redundancy

    
    # TODO
    def _get_support_mask(self):
        
        check_is_fitted(self, 'tcc_')
        
        return None
    
    
    """
    Helper function to recursively compute the body of a DecisionTreeClassifier
    """
    def _treebody2str(self, estimator, node_id, depth):

        children_left  = estimator.tree_.children_left
        children_right = estimator.tree_.children_right
        feature        = estimator.tree_.feature
        threshold      = estimator.tree_.threshold
        
        my_string = ""
        
        if children_left[node_id] == children_right[node_id]:
            
            # It is a leaf
            my_string = my_string + '%sreturn %s\n' % (' '*depth*4, estimator.classes_[np.argmax(estimator.tree_.value[node_id][0])])

        else:

            # Print the decision to take at this level
            my_string = my_string + '%sif X%d < %.3f:\n' % (' '*depth*4, (feature[node_id]+1), threshold[node_id])
            my_string = my_string + self._treebody2str(estimator, children_left[node_id],  depth+1)
            my_string = my_string + '%selse:\n' % (' '*depth*4)
            my_string = my_string + self._treebody2str(estimator, children_right[node_id], depth+1)
                
        return my_string

    
    """
    Convert a DecisionTreeClassifier into a string
    """
    def _DecisionTreeClassifier(self, estimator):

        # TODO: sanity check over estimator
        
        n_nodes        = estimator.tree_.node_count
        children_left  = estimator.tree_.children_left
        children_right = estimator.tree_.children_right
        feature        = estimator.tree_.feature

        tree_string = ""
        
        #
        # Compute the tree header
        #
        
        features_set = set()
                
        for node_id in range(n_nodes):

            # If we have a test node
            if (children_left[node_id] != children_right[node_id]):
                features_set.add('X%d' % (feature[node_id]+1))
        
        tree_string = tree_string + "def tree" + str(features_set) + ":\n"

        #
        # Compute the tree body
        # 
        
        tree_string = tree_string + self._treebody2str(estimator, 0, 1)

        return tree_string


    """
    Convert a MLPClassifier into a string
    """
    def _MLPClassifier(self, estimator):
        
        # TODO: sanity check over estimator
        
        # TODO: model string should not be based on numpy
        
        # Header
        string = "def NN(X):\n"
        
        # Parameters
        for i in np.arange(len(estimator.coefs_)):
            string = string + "    W" + str(i) + " = " + str(estimator.coefs_[i]) + "\n"
            string = string + "    b" + str(i) + " = " + str(estimator.intercepts_[i]) + "\n"
        
        # Computation
        for i in np.arange(len(estimator.coefs_) - 1):
            string = string + "    Z" + str(i) + " = np.matmul(W" + str(i) + ", X) + b" + str(i) + "\n"
            string = string + "    A" + str(i) + " = np.tanh(Z" + str(i) + ")\n"
        
        i = len(estimator.coefs_)
        string = string + "    Z" + str(i) + " = np.matmul(W" + str(i) + ", A" + str(i) + ") + b2\n"
        string = string + "    A" + str(i) + " = self._sigmoid(Z" + str(i) + ")\n"

        # Predictions        
        string = string + "    predictions = A" + str(i) + " > 0.5\n\n"
        
        string = string + "    return predictions\n"

        return string       

    """
    Convert a Keras MLP to a string representation
    """
    def _Keras_MLPClassifier(self, nn):
        # TODO: Review 
        # Header
        string = "def NN(X):\n"
            
        for i in np.arange(len(nn.layers)):
            if(not nn.layers[i].get_weights()): #In case of dropout layer. Skip. [check]
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
        
    
class Nescience(BaseEstimator, SelectorMixin):
    
    def __init__(self):
        
        return None
    
    
    def fit(self, X, y, method="Harmonic", compressor="bz2"):
        """
        Initialization of the class nescience
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Sample vectors from which to compute miscoding.
            
        y : array-like, shape (n_samples)
            The target values (class labels) as numbers or strings.

        method (string):     method used to comput the nescience. Valid
                             values are: "Euclid", "Arithmetic",
                             "Geometric", "Product", "Addition" and
                             "Harmonic".
                             
        compressor (string): compressor used to compute redudancy. Valid
                             values are: "bz2", "lzma" and "zlib".
          
        """
        
        self.method       = method
        self.compressor   = compressor

        self.X, self.y = check_X_y(X, y, dtype=None)

        self.miscoding  = Miscoding()
        self.inaccuracy = Inaccuracy()
        self.surfeit    = Surfeit()
        
        self.miscoding.fit(X, y)
        self.inaccuracy.fit(X, y)
        self.surfeit.fit(self.compressor)
        
        return self


    def nescience(self, model, subset=None, predictions=None, model_string=None):
        """
        Compute the nescience of a model
        
        Parameters
        ----------
        model       : a trained model

        subset      : array-like, shape (n_features)
                      1 if the attribute is in use, 0 otherwise
                      If None, attributes will be infrerred throught model
                      
        model_str   : a string based representation of the model
                      If None, string will be derived from model
                    
        Returns
        -------
        Return the nescience (float)
        """
        
        check_is_fitted(self, 'X')
        if subset == None:
            miscoding = self.miscoding.miscoding_model(model)
        else:
            miscoding = self.miscoding.miscoding_subset(subset)

        if predictions == None:
            inaccuracy = self.inaccuracy.inaccuracy_model(model)
        else:
            inaccuracy = self.inaccuracy.inaccuracy_predictions(predictions)
            
        if model_string == None:
            surfeit = self.surfeit.surfeit_model(model)
        else:
            surfeit = self.surfeit.surfeit_string(model_string)            

        # Avoid dividing by zero
        
        if surfeit == 0:
            surfeit = 10e-6
    
        if inaccuracy == 0:
            inaccuracy = 10e-6

        if miscoding == 0:
            miscoding = 10e-6

        # Compute the nescience according to the method specified by the user
        if self.method == "Euclid":
            # Euclidean distance
            nescience = math.sqrt(miscoding**2 + inaccuracy**2 + surfeit**2)
        elif self.method == "Arithmetic":
            # Arithmetic mean
            nescience = (miscoding + inaccuracy + surfeit) / 3
        elif self.method == "Geometric":
            # Geometric mean
            nescience = math.pow(miscoding * inaccuracy * surfeit, 1/3)
        elif self.method == "Product":
            # The product of both quantities
            nescience = miscoding * inaccuracy * surfeit
        elif self.method == "Addition":
            # The product of both quantities
            nescience = miscoding + inaccuracy + surfeit
        elif self.method == "Weighted":
            # Weigthed sum
            nescience = self.weight_miscoding * miscoding + self.weight_inaccuracy * inaccuracy + self.weight_surfeit * surfeit
        elif self.method == "Harmonic":
            # Harmonic mean
            nescience = 3 / ( (1/miscoding) + (1/inaccuracy) + (1/surfeit))
        # else -> rise exception
            
        #Modification for storing later in algorithm VALS array:
        #instead of: return nescience, 
        
        return nescience, surfeit, inaccuracy, miscoding
    
    # TODO
    def _get_support_mask(self):
                
        return None

