'''
Contains retrieval and evaluation functions for running after
algorithm execution finishes.
'''

from sklearn.datasets import load_digits, load_iris
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.models import load_model

import numpy as np
import pandas as pd
import pickle, sys, os, logging
import tensorflow as tf
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel(logging.ERROR)
tf.logging.set_verbosity(tf.logging.ERROR)

def get_classification_report(model, s_mean, s_var, viu, dataset):
    '''
    Given a model, its scaling parameters and viu and dataset, 
    generates classification reports and print general test accuracy.
    '''

    if "fmnist" in dataset:
        ((X_used, y_used), (X_test, y_test)) = fashion_mnist.load_data()
        X_test = X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2]) #flatten array
    elif "pulsars" in dataset:
        data = pd.read_csv("./datasets/pulsar_stars.csv")
        y=data.target_class
        X=data.drop("target_class", axis=1).values
    elif "digits" in dataset:
        data = load_digits()
        X = data.data
        y = data.target
    elif "magic" in dataset:
        data = pd.read_csv("./datasets/magic04.data", names=["fLength", "fWidth", "fSize", "fConc","fConc1","fAsym","fM3Long","fM3Trans","fAlpha","fDist","class"])
        y = (data['class'] == 'h').astype(int)
        X = data.drop('class', 1)
    elif "iris" in dataset:
        data = load_iris()
        X = data.data
        y = data.target
    else: 
        print("Unknown dataset selected.")
        return(1) 
    

    if "fmnist" not in dataset:
        X, X_test, y, y_test  = train_test_split(X,y, random_state = 42,test_size = 0.25)

    X_test = (X_test - s_mean) / np.sqrt(s_var)
   
    msdX_test = X_test[:, np.where(viu)[0]]

    y_test = to_categorical(y_test)
    y_pred = model.predict(msdX_test)

    y_test_t = np.argmax(y_test,axis=1)
    y_pred_t = np.argmax(y_pred,axis=1)

    print(classification_report(y_test_t, y_pred_t))
    print("Test accuracy:",model.evaluate(x=msdX_test, y=y_test)[1])

    return

def retrieve_results(runtime):
    '''
    Given runtime of execution of algorithm (In analysis/ folder)
    returns full trained model as well as viu, stat file and scaling params.
    '''

    with open("./analysis/"+runtime+"/stats.csv", 'r') as f:
        stats = pd.read_csv(f,names=["ind","numvius","Miscoding","Inaccuracy","Surfeit","Nescience"])

    net_id = stats['Nescience'].idxmin()
    num_vius = int(stats.iloc[net_id]["numvius"])

    model = load_model("./analysis/"+runtime+"/networks/Net"+str(net_id)+"_fullmodel.hdf5")
    model.summary()

    with open("./analysis/"+runtime+"/vius.txt", 'rb') as f:
        vius = pickle.load(f)
    viu = vius[num_vius-1]
    with open("./analysis/"+runtime+"/scaling_factors.txt", 'rb') as f:
        (s_mean, s_var) = pickle.load(f)
    

    return(model, viu, s_mean, s_var, stats)


def main():

    dataset = sys.argv[1]
    runtime = sys.argv[2]

    best_model, viu, s_mean, s_var, stats = retrieve_results(runtime)
    get_classification_report(best_model, s_mean, s_var, viu, dataset)

    exit(0)

main()