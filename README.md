**#Nescience Neural Network Classifier [WIP]**

There are two main versions of Nescience Neural Network Classifier:

+ NNNClassifier: Version using sequential (unoptimized) search

+ NNC-GE: Uses Grammatical Evolution (PonyGE2 algorithm) for developing the networks in a genetic-based way, focusing on minimizing the Nescience value on each generation.

***Usage of NNNC-GE Version***:

(Inside e.g Google Colab notebook)
Look at template: https://colab.research.google.com/drive/10s6wj6UG_2idI07S5EUwCzRonLkfw_rb

1. (if not loading directly .zip of PonyGE2 folder): Chdir inside NNC-GE/PonyGE2/src folder.
2. Select dataset in dropdown
3. Execute algorithm
4. Retrieve resulting best model
5. Evaluate the model