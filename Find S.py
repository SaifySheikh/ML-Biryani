
from google.colab import drive
drive.mount('/content/gdrive')

import pandas as pd
import numpy as np

data = pd.read_csv('/content/gdrive/MyDrive/ENJOYSPORT.csv')
data

data.head()

from matplotlib import pyplot as plt
import seaborn as sns
figsize = (12, 1.2 * len(_df_19['Humidity'].unique()))
plt.figure(figsize=figsize)
sns.violinplot(_df_19, x='index', y='Humidity', inner='stick', palette='Dark2')
sns.despine(top=True, right=True, bottom=True, left=True)

# @title EnjoySport

from matplotlib import pyplot as plt
data['EnjoySport'].plot(kind='hist', bins=20, title='EnjoySport')
plt.gca().spines[['top', 'right',]].set_visible(False)



# @title Water

from matplotlib import pyplot as plt
import seaborn as sns
data.groupby('Water').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
plt.gca().spines[['top', 'right',]].set_visible(False)

data.keys()

print(data.loc[0])

cols = len(data.keys()) - 1
cols

rows = len(data)
rows

concepts = np.array(data)[:,:-1]
concepts

target = np.array(data)[:,-1]
target

def train(concept, target):

  for i, val in enumerate(target):
    if(val == 1):
      specific_h = concept[i].copy()
      break

  for i, val in enumerate(concept):
    if(target[i] == 1):
      for j in range(len(specific_h)):
        if(val[j] != specific_h[j]):
          specific_h[j] = "?"
        else:
          pass

  return specific_h

train(concepts, target)