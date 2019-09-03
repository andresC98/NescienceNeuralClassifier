**#Nescience Neural Network Classifier [WIP]**

There are two main versions of Nescience Neural Network Classifier:

+ NNNClassifier: Version using sequential (unoptimized) search. 

+ NNC-GE: Uses Grammatical Evolution (PonyGE2 algorithm) for developing the networks in a genetic-based way, focusing on minimizing the Nescience value on each generation.

**Folder structure:**

- NNC-GE/PonyGE2: GE Version of Nescience NNC. Relevant files:
    - `grammars/mlpGrammar.pybnf`: Contains model grammars in Backus-Naur Form
    - `parameters/mlpConfig.txt`: Config file setting Genetic Algorithm parameters (num generations, population size...)
    - `src/fitness/NescNNclasGE.py`: Contains main evaluation (fitness) and Nescience functions.
    - `src/ponyge.py`: Algorithm execution script.
    - `src/datasets.py`: Contains datasets and internal load functions
    - `src/usedata.py`: Dataset selection script. Creates usedata.txt file for internal use.
    - `src/nnncutils.py`: Contains Retrieval and Evaluation functions for post-run analysis

**Usage of NNNC-GE Version:**

- ***Inside e.g Google Colab notebook (Free GPU):*** **(all libraries included in Colab env)**
    - Look at template: https://colab.research.google.com/drive/1pJOq9CmIe7eCgdVlpB01-PFMBAzdry8F

- ***As a script:*** **(some libraries may be required)**
    
    (Inside PonyGE2/src folder) 
    
    - 1.[Selecting dataset]: `python usedata.py <dataset>`
    - 2.[Algorithm execution]: `python ponyge.py --verbose --parameters mlpConfig.txt --generations 10 --population_size 15`
    - 3.[Results]: `python nnncutils.py <dataset> <runtime>`


 >Where \<dataset\> is one of the following: [digits, fmnist, magic, pulsars, iris]
    and \<runtime\> is algorithm exec time (same format as in analysis/ folder) "hh:mm:ss"

**Results evaluation:**

The algorithm will generate and store every individual (network) in .h5 format inside `analysis/<runtime>/networks` folder, so that it can be later retrieved 
using keras `load_model()` function, obtaining the full structure with its weights resulted from training. The best 
network (achieving lowest Nescience) will be also stored and can be recovered for later use loading its .h5 file identified
with its individual number (shown in algorithm).

To ease a quick evaluation of the results, the `nnncutils.py` script retrieves the model and evaluates it on the test set.

> Note: All networks and parameter files resulting from the algorithm execution will be stored inside the `src/analysis/<runtime>/` folder.

`analysis/<runtime>/stats.csv` file contains a dataframe that stores each evaluated individual's Nescience parameters.