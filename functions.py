import numpy as np;
import pandas as pd;
import math;



### Zamienia brakujące wartości cech - oznaczanych jako '?' - na najczęściej występującą wartość (w obrębie pytania)

def replaceMissingValues(data,labels):
  labelCount = 0;
  
  for column in data:
    currentFeatureName = labels[labelCount]; ### nazwa kolumny (cechy)
    currentFeatureVals = data[column];   ### wartości 
    unique = currentFeatureVals.unique() ### unikalne wartości cech
    #print("Unique values are: ", unique);
    if '?' in unique:
      #print("missing data encountered!");
      mostFreq = currentFeatureVals.mode();
      
      currentFeatureVals[currentFeatureVals == '?'] = mostFreq.item(); # zamiana 
    labelCount = labelCount+1;
  return data;
    

    
  ### Oblicza wartość średniej informacji (Average Entropy Information) zawartej w cesze
def avgEntropyInf(featEntropies,positives,negatives,globPosNeg):
  result = 0; 
  for e,p,n in zip(featEntropies,positives,negatives):
    #print(e);
    result = result + ((p+n)/(globPosNeg[0]+globPosNeg[1]))*e;
  #print(result);
  return result;


### Oblicza zysk informacji (Information Gain) przy podziale zbioru na danej cesze o atrybucie avgEntInf 
def infGain(entropy,avgEntInf):
  r = entropy - avgEntInf;
  #print(r);
  return r;
  
     


### Oblicza entropię zadanego zbioru z defiicji na podstasie ilości pozytywnych i negatywnych próbek
def calculateEntropy(positiveSamples, negativeSamples):
    if positiveSamples == 0 or negativeSamples == 0:
        entropyVal = 0;
    else:
        allSamples = positiveSamples + negativeSamples;
        entropyVal = - positiveSamples/allSamples * math.log2(positiveSamples/allSamples) - negativeSamples/allSamples * math.log2(negativeSamples/allSamples);
    return entropyVal;

### Zwraca entropię zbioru
def setEntropy(data):
  positiveSamples = 0; #positives--> e (jadalne)
  negativeSamples = 0; #neagatives --> p (trujące)
  entropyVal = 0;

  ### p - trujący e - jadalny
  #print("LABELS OVERLOOK {p - poisonous ; e - eatable} \n");

  for x in data.loc[:,"class"]:
   if x == 'p':
    negativeSamples = negativeSamples + 1;   
   elif x == 'e':
    positiveSamples = positiveSamples + 1;
   else:
    print("class not defined");

  #print("NUMBER OF POSITIVE (eatable) samples: ",positiveSamples);
  #print("NUMBER OF NEGATIVE (poisonous) samples: ",negativeSamples);
  
  entropyVal = calculateEntropy(positiveSamples, negativeSamples)
  return entropyVal;

### Oblicza entropię poszczególnych cech zbioru 'data'
def featureEntropy(data):
  globPosNeg = [0,0];
  gain = pd.DataFrame(columns=['Attribute', 'Gain']);
  for column in data: ### iteracja po kolumnach
    entrArr = []; 
    posArr = [];
    negArr = [];
    currentFeatureVals = data[column]; 
    unique = currentFeatureVals.unique()
    for x in range(unique.size): ### iteracja po atrybutach
      entropyFeat = 0;
      v = currentFeatureVals.loc[currentFeatureVals == unique[x]];
      #print(v)
      thisPositive = v.loc[data['class'] == 'e'].count();
      #print(thisPositive)
      thisNegative = v.loc[data['class'] == 'p'].count();
      #print(thisNegative)
      if column == 'class':
            if thisPositive !=0:
                globPosNeg[0] = thisPositive;
            if thisNegative !=0:
                globPosNeg[1] = thisNegative;
      #print(globPosNeg)
      entropyFeat = calculateEntropy(thisPositive,thisNegative);
      entrArr.append(entropyFeat);
      posArr.append(thisPositive);
      negArr.append(thisNegative);
    #print(globPosNeg)
    AEI = avgEntropyInf(entrArr,posArr,negArr,globPosNeg);

    gain = gain.append({'Attribute':column,'Gain':infGain(setEntropy(data),AEI)},ignore_index = True);
  return gain;

    
### Tworzy drzewo klasyfikujace za pomoca algorytmu ID3
def buildTree(df,tree=None):
    # Poszukiwanie atrybutu o najwiekszym zysku informacji
    fE = featureEntropy(df)
    fE = fE.drop(fE[fE.Attribute == 'class'].index)
    maxi = fE[fE['Gain']==fE['Gain'].max()] 
    node = maxi.iloc[0][0]

    if tree is None:
        tree = {}
        tree[node] = {}
        
    attValues = np.unique(df[node])
    # Budowanie drzewa na podstawie wartosci wyznaczonego atrybutu
    for value in attValues:
        subtable = df[df[node] == value].reset_index(drop=True)
        subtable = subtable.drop(node, 1)
        counts = subtable['class'].value_counts()
        # Sprawdzanie czy zachodzi kryterium stopu
        if len(counts)==1:
            tree[node][value] = counts.index.values[0]
        elif len(subtable.columns) == 1 or ~(0.01 <= (counts.values[0]/(counts.values[0]+counts.values[1])) <= 0.99):
            if counts.values[0] > counts.values[1]:
                tree[node][value] = counts.index.values[0]
            else:
                tree[node][value] = counts.index.values[1]
        else:    # Rozrastanie drzewa    
            tree[node][value] = buildTree(subtable)
    return tree

### Tworzy drzewo klasyfikujace za pomoca zmodyfikowanego algorytmu ID3 
def buildTree_deeper(df,tree=None):
    # Poszukiwanie atrybutu o najwiekszym zysku informacji
    fE = featureEntropy(df)
    fE = fE.drop(fE[fE.Attribute == 'class'].index)
    max_gain = 0
    node = None
    # Wyznaczanie zysku informacji dla kazdego z poddrzew
    for i in range(len(fE)):
        feat = fE.loc[i+1][0]
        root_gain = fE.loc[i+1][1]
        attValues = np.unique(df[feat])
        nodes_gain = []
        for value in attValues:
            subtable = df[df[feat] == value].reset_index(drop=True)
            if len(subtable['class'].value_counts()) == 1:
                nodes_gain.append(1)
            else:
                fE_subtable = featureEntropy(subtable)
                fE_subtable = fE_subtable.drop(fE_subtable[fE_subtable.Attribute == 'class'].index)
                maxi_subtable = fE_subtable[fE_subtable['Gain']==fE_subtable['Gain'].max()]
                nodes_gain.append(maxi_subtable.iloc[0][1])
        gain = root_gain + sum(nodes_gain)/len(nodes_gain)
        if gain > max_gain:
            max_gain = gain
            node = feat
             
    if tree is None:
        tree = {}
        tree[node] = {}
    # Budowanie drzewa na podstawie wartosci wyznaczonego atrybutu    
    attValues = np.unique(df[node])
    for value in attValues:
        subtable = df[df[node] == value].reset_index(drop=True)
        subtable = subtable.drop(node, 1)
        counts = subtable['class'].value_counts()
        if len(counts)==1:
            tree[node][value] = counts.index.values[0]
        elif len(subtable.columns) == 1 or ~(0.01 <= (counts.values[0]/(counts.values[0]+counts.values[1])) <= 0.99):
            if counts.values[0] > counts.values[1]:
                tree[node][value] = counts.index.values[0]
            else:
                tree[node][value] = counts.index.values[1]
        else:        
            tree[node][value] = buildTree_deeper(subtable)
    return tree

### Funkcja wspomagajaca testowanie, ktora pozwala na zaglebianie sie
def tree_go_deeper(tree, df):
    key = next(iter(tree.keys()))
    val = df[key]
    new_tree = tree[key][val]
    return new_tree

### Funkcja testujaca dane drzewo i zwracajaca macierz pomylek
def test_tree(tree,df):
    conf_matrix = [0,0,0,0] #[TP, FP, TN, FN]
    for i in range(len(df)):
        row = df.loc[i]
        new_tree = tree

        while isinstance(new_tree,dict):
            new_tree = tree_go_deeper(new_tree,row)
        real_class = row['class']
        if new_tree == 'e' and real_class =='e':
            conf_matrix[0] += 1
        elif new_tree == 'e' and real_class =='p':
            conf_matrix[1] += 1
        elif new_tree == 'p' and real_class =='p':
            conf_matrix[2] += 1
        else:
            conf_matrix[3] += 1
    return conf_matrix
                
    