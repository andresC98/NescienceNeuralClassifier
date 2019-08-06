**#Nescience Neural Network Classifier [WIP]**

There are two main versions of Nescience Neural Network Classifier:

+ NNNClassifier: Version using sequential (unoptimized) search

+ NNC-GE: Uses Grammatical Evolution (PonyGE2 algorithm) for developing the networks in a genetic-based way, focusing on minimizing the Nescience value on each generation.

**Folder structure:**

- NNC-GE/PonyGE2: GE Version of Nescience NNC. Relevant files:
    - `grammars/mlpConfig.txt`: Contains model grammars in Backus-Naur Form
    - `src/fitness/NescNNclasGE.py`: Contains main evaluation (fitness) and Nescience functions.
    - `src/ponyge.py`: Algorithm execution script.
    - `src/datasets.py`: Contains datasets and internal load functions
    - `src/usedata.py`: Dataset selection script. Creates usedata.txt file for internal use.
    - `src/nnncutils.py`: Contains Retrieval and Evaluation functions for post-run analysis

**Usage of NNNC-GE Version:**

- ***Inside e.g Google Colab notebook (Free GPU):*** **(all libraries included in Colab env)**
    - Look at template: https://colab.research.google.com/drive/1YTooVm6TIiWHvwx4YkEB9xZM_PZzkEOW

- ***As a script:*** **(some libraries may be required)**
    
    (Inside PonyGE2/src folder) 
    
    - [Selecting dataset]: `python usedata.py <dataset>`
    - [Algorithm execution]: `python ponyge.py --verbose --parameters mlpConfig.txt --generations 10 --population_size 15`
    - [Results]: `python nnncutils.py <dataset> <runtime>`


 >Where \<dataset\> is one of the following: [digits, fmnist, magic, pulsars, iris]
    and \<runtime\> is algorithm exec time (same format as in analysis/ folder) "hh:mm:ss"