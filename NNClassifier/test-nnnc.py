from NNNCv2 import *
import sys

def main():

    if(sys.argv[1] in "digits"):
        from sklearn.datasets import load_digits
        data = load_digits()
        X = data.data
        y = data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    elif(sys.argv[1] in "iris"):

        # from numpy import genfromtxt
        # import pandas as pd
        # data = pd.read_csv('Iris.csv')
        # data = data.drop('Id',axis = 1)
        # train, test = train_test_split(data, test_size = 0.33)
        # X_train = train[['SepalLengthCm', 'SepalWidthCm','PetalLengthCm','PetalWidthCm']]
        # y_train = train.Species
        # X_test = test[['SepalLengthCm', 'SepalWidthCm','PetalLengthCm','PetalWidthCm']]
        # y_test = test.Species
        # y_train = y_train.astype("category").cat.codes
        # y_test = y_test.astype("category").cat.codes   

        from sklearn.datasets import load_iris
        data = load_iris()
        X = data.data
        y = data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    elif(sys.argv[1] in "pulsar"):
        from numpy import genfromtxt
        data = genfromtxt('pulsar_stars.csv', delimiter=',')
        data = np.delete(data, 0, 0)
        X = data[:,:8]
        y = data[:,8]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    else:
        sys.exit()

    print("X train :{}. y_Train: {}, X test: {}, y_test: {}".format(X_train.shape,y_train.shape, X_test.shape, y_test.shape))

    model = NescienceNeuralNetworkClassifier(verbose = True)
    model.fit(X_train, y_train)
    model.get_model_scores(X_test, y_test)

main()