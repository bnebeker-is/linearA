import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.chdir("C:/Users/brett.nebeker/Documents/Personal Docs/Joel")

df = pd.read_csv("./dropbox/SmallDeptList_small_for_agg.csv")
df.head()
print(df.shape)

## isolate single dept
df = df[df.DEPARTMENT_ID == 102011]

## aggregation by date
df = df.groupby('VISIT_DTM').count().reset_index()

df.loc[:,'VISIT_DTM'] = pd.to_datetime(df.loc[:,'VISIT_DTM'])

df.plot()
plt.show()
