import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet

os.chdir("C:/Users/brett.nebeker/Documents/Personal Docs/Joel")

df = pd.read_csv("./dropbox/SmallDeptList_small_for_agg.csv")
df.head()
print(df.shape)

## isolate single dept
df = df[df.DEPARTMENT_ID == 102011]

## aggregation by date
df = df.groupby('VISIT_DTM').count().reset_index()

df.loc[:, 'VISIT_DTM'] = pd.to_datetime(df.loc[:, 'VISIT_DTM'])
df.columns = ['ds', 'y']

plt.plot(df.ds, df.y)

df.ds.describe()
df_train = df[df.ds < '2017-01-01']
df_test = df[df.ds >= '2017-01-01']

m = Prophet()
m.fit(df_train)

future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

forecast = forecast.merge(df)

model_eval = forecast[['ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper']]

fig1 = m.plot(model_eval)

model_eval.loc[:, 'in_range'] = np.where((model_eval.loc[:, 'y'] >= model_eval.loc[:, 'yhat_lower']) &
                                         (model_eval.loc[:, 'y'] <= model_eval.loc[:, 'yhat_upper']), 1, 0)

test = model_eval[model_eval.ds >= '2017-01-01']
## % in range
print(test.in_range.mean())

## mean abs % error
MAPE = np.mean(np.abs((test.loc[:, 'y'] - test.loc[:, 'yhat']) / test.loc[:, 'y'])) * 100
print(MAPE)

