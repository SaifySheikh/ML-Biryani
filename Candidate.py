
from google.colab import drive
drive.mount('/content/gdrive')

import pandas as pd
import numpy as np

data = pd.read_csv('/content/gdrive/MyDrive/Machine Learning/ENJOYSPORT.csv')
data

data.head()

data.keys()

concepts = np.array(data)[:,:-1]
concepts

target = np.array(data)[:,-1]
target

def learn(concepts, target):

    specific_h = concepts[0].copy()
    print("initialization of specific_h and general_h")
    print(specific_h)
    general_h = [["?" for i in range(len(specific_h))] for i in range(len(specific_h))]
    print(general_h)

    for i, h in enumerate(concepts):



        if target[i] == 1:
            for x in range(len(specific_h)):
                if h[x]!= specific_h[x]:
                    specific_h[x] ='?'
                    general_h[x][x] ='?'
                print(specific_h)
        print(specific_h)



        if target[i] == 0:
            for x in range(len(specific_h)):
                if h[x]!= specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'


        print(" steps of Candidate Elimination Algorithm",i+1)
        print(specific_h)
        print(general_h)



    indices = [i for i, val in enumerate(general_h) if val == ['?', '?', '?', '?', '?', '?']]

    for i in indices:
        general_h.remove(['?', '?', '?', '?', '?', '?'])

    return specific_h, general_h

s_final, g_final = learn(concepts, target)

print("Final Specific_h : ", s_final, sep="\n")
print("Final General_h : ", g_final, sep="\n")

