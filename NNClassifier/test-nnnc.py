from NNNCv2 import *
from numpy import genfromtxt
from sklearn.datasets import load_digits, load_iris, load_breast_cancer
from keras.datasets import mnist
import pandas as pd
import sys
import matplotlib.pyplot as plt

def main():

    if(len(sys.argv) > 1):
        if(sys.argv[1] in "--h"):
            print("Options:\n\t> digits/iris/pulsar/bank/mnist: Loads with selected dataset.\n\t> -p: Plot search result after finishing algorithm.")
            sys.exit()
        else:
            if(sys.argv[1] in "digits"):
                data = load_digits()
                X = data.data
                X = (X - np.min(X)) / (np.max(X) - np.min(X))
                y = data.target
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
            elif(sys.argv[1] in "iris"):
                data = load_iris()
                X = data.data
                X = (X - np.min(X)) / (np.max(X) - np.min(X))
                y = data.target
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
            elif(sys.argv[1] in "pulsar"):
                data = genfromtxt('./data/pulsar_stars.csv', delimiter=',')
                data = np.delete(data, 0, 0)
                X = data[:,:8]
                X = (X - np.min(X)) /   (np.max(X) - np.min(X))
                y = data[:,8]   
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
            elif(sys.argv[1] in "bank"):
                data = genfromtxt('./data/data_banknote_authentication.txt', delimiter=',')
                X = data[:,:4]
                X = (X - np.min(X)) / (np.max(X) - np.min(X))
                y = data[:,4]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
            elif(sys.argv[1] in "mnist"):
                (X_train, y_train), (X_test, y_test) = mnist.load_data()
                X_train = (X_train - np.min(X_train)) / (np.max(X_train) - np.min(X_train))
                X_test = (X_test - np.min(X_test)) / (np.max(X_test) - np.min(X_test))
            elif(sys.argv[1] in "breast"):
                data = load_breast_cancer()
                X = data.data
                y = data.target
                X = (X - np.min(X)) / (np.max(X) - np.min(X))
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
            elif(sys.argv[1] in "sonar"):
                data = pd.read_csv('./data/sonar.all-data.csv')
                X = data.drop('R', axis=1)
                y = pd.factorize(data.R)[0]
                X = np.array(X)
                y = np.array(y)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
            else:
                sys.exit()
            
    else:
        print("Wrong usage.")
        print("Options:\n\t> digits/iris/pulsar/bank/mnist: Loads with selected dataset.\n\t> -p: Plot search result after finishing algorithm.")
        sys.exit()

    print("X train :{}. y_Train: {}, X test: {}, y_test: {}".format(X_train.shape,y_train.shape, X_test.shape, y_test.shape))
    
    if(len(sys.argv)  > 3 and "-s" in sys.argv[2]):
        stop = int(sys.argv[3])
    elif(len(sys.argv)  > 4 and "-s" in sys.argv[3]):
        stop = int(sys.argv[4])
    else:
        stop = 0

    model = NescienceNeuralNetworkClassifier(verbose = True)
    model.fit(X_train, y_train, run_until= stop)
    model.get_model_scores(X_test, y_test)


    if("-p" in sys.argv): #autoplot option on
        model_h = model.get_model_hist()
        model.vis_nescience(model_h)
        plt.show(block=True)


main()