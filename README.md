**#Nescience Neural Network Classifier [WIP]**

There are two main versions of Nescience Neural Network Classifier:

+ NNNClassifier: Version using sequential (unoptimized) search

+ NNC-GE: Uses Grammatical Evolution (PonyGE2 algorithm) for developing the networks in a genetic-based way, focusing on minimizing the Nescience value on each generation.

***Usage of NNNC-GE Version***:

***+ (Inside e.g Google Colab notebook)*** (all libraries included in Colab env)
Look at template: https://colab.research.google.com/drive/1YTooVm6TIiWHvwx4YkEB9xZM_PZzkEOW

***+ (As a script)*** (some libraries may be required)
    (Inside PonyGE2/src folder) 
    1. > python usedata.py <dataset>
    1. > python ponyge.py --verbose --parameters mlpConfig.txt --generations 10 --population_size 15
    2. > python nnncutils.py <dataset> <runtime>
    
    Where <dataset> is one of the following: [digits, fmnist, magic, pulsars, iris]
    and <runtime> is algorithm exec time (same format as in analysis/ folder) "hh:mm:ss"