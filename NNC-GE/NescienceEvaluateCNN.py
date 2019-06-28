import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import nescience
from nescience.Nescience import Nescience

from fitness.base_ff_classes.base_ff import base_ff

class NescienceEvaluateCNN(base_ff):
    def __init__(self):
        #initialize base_ff class
        super().__init__()

        #Initialize hyperparameters and prepare the data

        self.batch_size = 128
        self.num_classes = 10
        self.epochs = 1

        # input image dimensions
        self.img_rows, self.img_cols = 28, 28

        # the data, split between train and test sets
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

        if K.image_data_format() == 'channels_first':
            self.x_train = self.x_train.reshape(self.x_train.shape[0], 1, self.img_rows, self.img_cols)
            self.x_test = self.x_test.reshape(self.x_test.shape[0], 1, self.img_rows, self.img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            self.x_train = self.x_train.reshape(self.x_train.shape[0], self.img_rows, self.img_cols, 1)
            self.x_test = self.x_test.reshape(self.x_test.shape[0], self.img_rows, self.img_cols, 1)
            input_shape = (self.img_rows, self.img_cols, 1)

        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        self.x_train /= 255
        self.x_test /= 255

        self.x_train.astype(int)
        self.x_test.astype(int)
        self.y_train.astype(int)
        self.y_test.astype(int)

        # convert class vectors to binary class matrices
        self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes)
        self.y_test = keras.utils.to_categorical(self.y_test, self.num_classes)
        self.y_train.reshape((self.y_train.shape[0], self.y_train.shape[1]))
        self.y_test.reshape((self.y_test.shape[0], self.y_test.shape[1]))
        print('data type:', type(self.x_test[0][0][0]))


        #K.set_image_dim_ordering('th')

    def evaluate(self, ind, **kwargs):
        
        model = eval(ind.phenotype)
        print('LAYERS ARE HERE:', model.layers)
        model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
        
        self.y_test = np.reshape(self.y_test, (self.y_test.shape[0], self.y_test.shape[1]))

        model.fit(self.x_train, self.y_train,
          batch_size=self.batch_size,
          epochs=self.epochs,
          verbose=1,
          validation_data=(self.x_test, self.y_test))

        module = __import__('nescience')
        class_ = getattr(module, 'Nescience')
        self.x_test = self.x_test.astype(int)
        self.y_test = self.y_test.astype(int)
        print(self.x_test.shape, 'XSHAPE')
        nes = Nescience(self.x_test.reshape((10000,self.img_cols*self.img_rows)), self.y_test, verbose = True)

        tcc = nes.targetconditionalcomplexity()
        attr_in_use = []
        for i in range(len(tcc)):
            if tcc[i] != 0:
                attr_in_use.append(1)
            else:
                attr_in_use.append(0)
        print(len(tcc), len(attr_in_use))

        errors = []
        predictions = model.predict(self.x_test, verbose=1).argmax(axis = 1)
        groundTruth = self.y_test.argmax(axis = 0)
        
        print("prediction shape:", predictions.shape)
        print("label shape:", self.y_test.shape)

        for elementIndex in range(len(groundTruth)):
            if groundTruth[elementIndex] != predictions[elementIndex]:
                errors.append(self.x_test[elementIndex])
        errors = np.array(errors).reshape((len(errors), self.img_cols*self.img_rows))
        print(np.array(errors).shape)
        return nes.nescience(attr_in_use, errors, self._cnn2str(model))

    
    def _cnn2str(self, nn):
        '''
        nn (keras model) - model to be made into string form
        '''
        
        layers = nn.layers
        
        string = "def CNN(X):\n"
        counter = 0
        for layer in layers:
            config = layer.get_config()

            if counter == 0:
                layerInput = 'X'
                string += '    depth = ' + layerInput + '.shape[0]\n' +\
                '    height = ' + layerInput + '.shape[1]\n' +\
                '    width = ' + layerInput + '.shape[2]\n' +\
                '    stride = ' + str(config['strides'][0]) + '\n' +\
                '    padding = 1\n' +\
                '    kernels = ' + str(layer.get_weights()[0][:,:,0,:]) + '\n' +\
                '    outputHeight = int((height - len(kernels) + 2*padding)/stride + 1)\n' +\
                '    outputWidth = int((width - len(kernels[0]) + 2*padding)/stride + 1)\n\n' + \
                '    output = []\n' +\
                '    counter = 0\n' +\
                '    for kernelIndex in range(kernels.shape[2]):\n' +\
                '        paddedX = np.zeros((depth,height+2*padding, width +2*padding))\n' +\
                '        for dep in range(depth):\n' +\
                '            for row in range(height):\n' +\
                '                for col in range(width):\n' +\
                '                    paddedX[dep][row+padding][col+padding] = X[dep][row][col]\n\n' +\
                '        kernel = kernels[:,:,kernelIndex]\n'+\
                '        rowRange = paddedX.shape[1] - kernel.shape[0] +1\n' +\
                '        colRange = paddedX.shape[2] - kernel.shape[1] +1\n' +\
                '        for layer in range(depth):\n'+\
                '            for row in range(0, rowRange, stride):\n' +\
                '                for col in range(0, colRange, stride):\n' +\
                '                    out = 0\n' +\
                '                    for xKernel in range(kernel.shape[0]):\n' +\
                '                        for yKernel in range(kernel.shape[1]):\n' +\
                '                            out += kernel[xKernel, yKernel]*paddedX[layer, row + xKernel, col + yKernel]\n' +\
                '                    if out < 0 and activation == \'relu\':\n' +\
                '                        out = 0\n' +\
                '                    output.append(out)\n' +\
                '                    counter += 1\n' +\
                '    X' + str(counter) +' = np.array(output)\n' +\
                '    X' + str(counter) +' = out.reshape((depth,kernels.shape[2],outputHeight, outputWidth))\n'
                counter += 1
                continue
            else:
                layerInput = 'X' + str(counter-1)
                    
            if 'conv' in config['name']:
                string += '    depth = ' + layerInput + '.shape[0]\n' +\
                '    height = ' + layerInput + '.shape[2]\n' +\
                '    width = ' + layerInput + '.shape[3]\n' +\
                '    stride = ' + str(config['strides'][0]) + '\n' +\
                '    kernelLength = ' + layerInput + '.shape[1]\n' +\
                '    padding = 1\n' +\
                '    kernels = ' + str(layer.get_weights()[0][:,:,0,:]) + '\n' +\
                '    outputHeight = int((height - len(kernels) + 2*padding)/stride + 1)\n' +\
                '    outputWidth = int((width - len(kernels[0]) + 2*padding)/stride + 1)\n\n' + \
                '    output = []\n' +\
                '    counter = 0\n' +\
                '    for kernelIndex in range(kernels.shape[2]):\n' +\
                '        paddedX = np.zeros((depth,height+2*padding, width +2*padding))\n' +\
                '        for dep in range(depth):\n' +\
                '            for row in range(height):\n' +\
                '                for col in range(width):\n' +\
                '                    paddedX[dep][row+padding][col+padding] = X[dep][row][col]\n\n' +\
                '        kernel = kernels[:,:,kernelIndex]\n'+\
                '        rowRange = paddedX.shape[1] - kernel.shape[0] +1\n' +\
                '        colRange = paddedX.shape[2] - kernel.shape[1] +1\n' +\
                '        for layer in range(depth):\n'+\
                '            for row in range(0, rowRange, stride):\n' +\
                '                for col in range(0, colRange, stride):\n' +\
                '                    out = 0\n' +\
                '                    for xKernel in range(kernel.shape[0]):\n' +\
                '                        for yKernel in range(kernel.shape[1]):\n' +\
                '                            out += kernel[xKernel, yKernel]*paddedX[layer, row + xKernel, col + yKernel]\n' +\
                '                    if out < 0 and activation == \'relu\':\n' +\
                '                        out = 0\n' +\
                '                    output.append(out)\n' +\
                '                    counter += 1\n' +\
                '    X' + str(counter) +' = np.array(output)\n' +\
                '    X' + str(counter) +' = out.reshape((depth,kernels.shape[2],outputHeight, outputWidth))\n'
                
            elif 'pooling' in config['name']:
                
                string += '    depth = ' + layerInput + '.shape[0]\n' +\
                '    height = ' + layerInput + '.shape[2]\n' +\
                '    width = ' + layerInput + '.shape[3]\n' +\
                '    stride = ' + str(config['strides'][0]) +\
                '\n    numKernels = ' + layerInput + '.shape[0]\n' +\
                '\n    poolSize = '+ str(config['pool_size'][0]) +\
                '\n\n    outputHeight = int((height - poolSize)/stride + 1)\n' +\
                '    outputWidth = int((width - size)/stride + 1)\n\n' +\
                '    output = []\n\n' +\
                '    for dep in range(depth):\n' +\
                '        for kernel in numKernels:\n'
                '            for row in range(0,height - poolSize + 1, stride):\n' +\
                '                for col in range(0, width - poolSize + 1, stride):\n' +\
                '                    out = 0\n' +\
                '                    if poolType == \'max\':\n' +\
                '                        for xKernel in range(poolSize):\n' +\
                '                            for yKernel in range(poolSize):\n' +\
                '                                out = max(out, ' + layerInput + '[dep, kernel, row + xKernel, col + yKernel])\n' +\
                '                    else:\n' +\
                '                        print(\'poolType not supported\')\n' +\
                '                        return\n' +\
                '                    output.append(out)\n\n' +\
                '    X' + str(counter) + ' = np.array(output)\n' +\
                '    X' + str(counter) + ' = np.reshape(X' + str(counter) + ',(depth,numKernels, outputHeight, outputWidth))\n'
                
            elif 'dropout' in config['name']:
                rate = config['rate']
                
                string += '    numDroped = int(' + str(rate) + ' * ' + layerInput + '.size\n' + \
                        '    toDrop = random.sample(range(0,' + layerInput + '.size), numDropped)\n'+\
                        '    originalShape = '+ layerInput + '.shape\n' +\
                        '    flat = ' + layerInput + '.flatten()\n' +\
                        '    for index in toDrop:\n' +\
                        '        flat[index] = 0\n' +\
                        '    X' + str(counter) +  ' = np.reshape(flat, originalShape)\n'
                
            elif 'dense' in config['name']:
                weights = layer.get_weights()
                string += '    W = ' + str(weights[0]) + '\n' +\
                '    b = ' + str(weights[1]) + '\n' +\
                '    X' + str(counter) +'= np.matmul(W, ' + layerInput + ') + b\n'
                
                if config['activation'] == 'relu':
                    string += '    negatives = np.where(X' + str(counter) +' < 0)\n'+\
                    '    for index in negatives:\n' +\
                    '        X' + str(counter) +'[index] = 0\n'                 
                elif config['activation'] == 'softmax':
                    string += '    sum = 0\n' +\
                            '    for element in X' + str(counter-1) + ':\n' +\
                            '        sum += np.exp(element)\n' +\
                            '    X' + str(counter) + ' = []\n' +\
                            '    for element in X' + str(counter - 1) + ':\n' +\
                            '        X' + str(counter) + '.append(exp(element)/sum)\n'
                else:
                    print('we do not support this dense layer activation')
                    return    
            else:
                print('layer ' + config['name'] + ' not supported')
            
            counter += 1
        
        
        string += '    predictions = X' + str(counter-1) + ' > 0.5\n' +\
                '    return predictions'
        return string
            
