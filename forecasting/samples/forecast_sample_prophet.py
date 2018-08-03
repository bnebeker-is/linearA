import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error, \
    explained_variance_score, mean_absolute_error, median_absolute_error
from sklearn.model_selection import train_test_split

# os.chdir("C:/Users/brett.nebeker/Documents/Personal Docs/Joel")

# df = pd.read_csv("C:/Users/brett.nebeker/Documents/Personal Docs/Joel/dropbox/SmallDeptList_small_for_agg.csv")
df = pd.read_csv("/home/brett/Downloads/SmallDeptList.csv")

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
train_df = df[df.ds <= '2017-10-01']
test_df = df[(df.ds > '2017-10-01') & (df.ds <= '2017-10-11')]

m = Prophet()
m.fit(train_df)

future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

forecast = forecast.merge(df)

model_eval = forecast[['ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper']]

fig1 = m.plot(model_eval)

model_eval.loc[:, 'in_range'] = np.where((model_eval.loc[:, 'y'] >= model_eval.loc[:, 'yhat_lower']) &
                                         (model_eval.loc[:, 'y'] <= model_eval.loc[:, 'yhat_upper']), 1, 0)

oos_df = model_eval[model_eval.ds >= '2017-01-01']
## % in range
print(oos_df.in_range.mean())

## mean abs % error
MAPE = np.mean(np.abs((oos_df.loc[:, 'y'] - oos_df.loc[:, 'yhat']) / oos_df.loc[:, 'y'])) * 100
print(MAPE)



# mean square error
mse = mean_squared_error(
    y_true=oos_df.loc[:, "y"],
    y_pred=oos_df.loc[:, "yhat"]
)
# root mean square error
rmse = np.sqrt(mse)

# mean absolute error
mae = mean_absolute_error(
    y_true=oos_df.loc[:, "y"],
    y_pred=oos_df.loc[:, "yhat"]
)

mdae = median_absolute_error(
    y_true=oos_df.loc[:, "y"],
    y_pred=oos_df.loc[:, "yhat"]
)

exp_var = explained_variance_score(
    y_true=oos_df.loc[:, "y"],
    y_pred=oos_df.loc[:, "yhat"]
)

print("RMSE: {0}".format(rmse))
print("MSE: {0}".format(mse))
print("MAE: {0}".format(mae))
print("MED. AE: {0}".format(mdae))
print("EXP VAR: {0}".format(exp_var))

# RMSE: 31.712468493564696
# MSE: 1005.6806579553335
# MAE: 23.33173729451341
# MED. AE: 18.648329232994314
# EXP VAR: 0.5161827631099696
