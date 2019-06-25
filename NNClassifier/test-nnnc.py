from NNNCv2 import *
from numpy import genfromtxt
from sklearn.datasets import load_digits, load_iris
import sys

def main():

    if(sys.argv[1] in "digits"):
        data = load_digits()
        X = data.data
        #X = (X - np.min(X)) / (np.max(X) - np.min(X))
        y = data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    elif(sys.argv[1] in "iris"):
        data = load_iris()
        X = data.data
        #X = (X - np.min(X)) / (np.max(X) - np.min(X))
        y = data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    elif(sys.argv[1] in "pulsar"):
        data = genfromtxt('pulsar_stars.csv', delimiter=',')
        data = np.delete(data, 0, 0)
        X = data[:,:8]
        #X = (X - np.min(X)) / (np.max(X) - np.min(X))
        y = data[:,8]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    elif(sys.argv[1] in "bank"):
        data = genfromtxt('data_banknote_authentication.txt', delimiter=',')
        X = data[:,:4]
        X = (X - np.min(X)) / (np.max(X) - np.min(X))
        y = data[:,4]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    else:
        sys.exit()

    print("X train :{}. y_Train: {}, X test: {}, y_test: {}".format(X_train.shape,y_train.shape, X_test.shape, y_test.shape))

    model = NescienceNeuralNetworkClassifier(verbose = True)
    model.fit(X_train, y_train, run_until = 5)
    model.get_model_scores(X_test, y_test)

main()