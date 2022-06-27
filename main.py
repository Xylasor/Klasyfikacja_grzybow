# AUTORZY:
# RAFAŁ J. TRYBUS, NR INDEKSU: 274356
# MATEUSZ HRYCIOW, NR INDEKSU: 283365
#--------ALGORYTM ID #3 ----------------------------------------------- #
import numpy as np;
import pandas as pd;
import pprint;
from functions import setEntropy;
from functions import calculateEntropy;
from functions import featureEntropy;
from functions import avgEntropyInf;
from functions import infGain;
from functions import buildTree;
from functions import buildTree_deeper;
from functions import tree_go_deeper;
from functions import test_tree;
from functions import replaceMissingValues;

### loading data from .data file
dataFile = open('agaricus-lepiota.data', 'r') 
mushrooms = dataFile.readlines() 



#------------------------------------
mushroom_labels = ['class','cap-shape','cap-surface','cap-color','bruises?','odor','gill-attachment','gill-spacing','gill-size','gill_color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat']
mushrooms = [x.replace('\n', '') for x in mushrooms]
mushrooms = pd.DataFrame([sub.split(",") for sub in mushrooms])
mushrooms.columns = mushroom_labels
print(mushrooms)

replaceMissingValues(mushrooms,mushroom_labels);
train=mushrooms.sample(frac=0.7,random_state=300)
test=mushrooms.drop(train.index)

tree1 = buildTree(train)
tree2 = buildTree_deeper(train)

results1 = test_tree(tree1,test.reset_index())
print('Wyniki w postaci macierzy pomylek [TP,FP,TN,FN]')
print('Dla algorytmu ID3')
print(results1)
print('Dla zmodyfikowanego algorytmu ID3')
results2 = test_tree(tree2,test.reset_index())
print(results2)

print('Dla algorytmu ID3 otrzymano drzewo')
pprint.pprint(tree1)
print('Dla zmodyfikowanego algorytmu ID3 otrzymano drzewo')
pprint.pprint(tree2)