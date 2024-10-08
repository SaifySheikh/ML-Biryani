
from google.colab import drive
drive.mount('/content/gdrive')

import pandas as pd
import numpy as np
from math import log2

df = pd.read_csv('/content/gdrive/MyDrive/play_tennis.csv')
df

y=0
n=0
j = len(df)
for i in df['play']:
  if i == 'Yes':
    y+=1
  else:
    n+=1
e_total = - (y/j)*log2(y/j) - (n/j)*log2(n/j)
print(round(e_total,2))






ans = ''
max = -999
for col in df.columns:
  if col != 'play' and col != 'day':
    sum = 0
    for attVal in df[col].unique():
      sy = 0
      sn = 0
      for i in range(len(df)):
        if df[col][i] == attVal:
          if df['play'][i] == 'Yes':
            sy+=1
          else:
            sn+=1
      t = sy + sn
      if sy !=0 and sn != 0:
        en = - (sy/t)*log2(sy/t) - (sn/t)*log2(sn/t)
        sum += ((t/len(df)) * en)

    gain = e_total - round(sum,2)
    print(f"Information Gain for {col} = {gain}")
    if gain > max:
      ans = col
      max = gain
print("\n"+ans +'\tis root node')

