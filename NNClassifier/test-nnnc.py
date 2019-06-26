from NNNCv2 import *
from numpy import genfromtxt
from sklearn.datasets import load_digits, load_iris
from keras.datasets import mnist
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
                data = genfromtxt('pulsar_stars.csv', delimiter=',')
                data = np.delete(data, 0, 0)
                X = data[:,:8]
                X = (X - np.min(X)) /   (np.max(X) - np.min(X))
                y = data[:,8]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
            elif(sys.argv[1] in "bank"):
                data = genfromtxt('data_banknote_authentication.txt', delimiter=',')
                X = data[:,:4]
                X = (X - np.min(X)) / (np.max(X) - np.min(X))
                y = data[:,4]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
            elif(sys.argv[1] in "mnist"):
                (X_train, y_train), (X_test, y_test) = mnist.load_data()
                X_train = (X_train - np.min(X_train)) / (np.max(X_train) - np.min(X_train))
                X_test = (X_test - np.min(X_test)) / (np.max(X_test) - np.min(X_test))
            else:
                sys.exit()
    else:
        print("Wrong usage.")
        print("Options:\n\t> digits/iris/pulsar/bank/mnist: Loads with selected dataset.\n\t> -p: Plot search result after finishing algorithm.")
        sys.exit()

    print("X train :{}. y_Train: {}, X test: {}, y_test: {}".format(X_train.shape,y_train.shape, X_test.shape, y_test.shape))

    model = NescienceNeuralNetworkClassifier(verbose = True)
    model.fit(X_train, y_train)
    model.get_model_scores(X_test, y_test)

    if(len(sys.argv) == 3 and sys.argv[2] in "-p"): #autoplot option on
        model_h = model.get_model_hist()
        model.vis_nescience(model_h)
        plt.show(block=True)


main()