import collections
import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, \
    explained_variance_score, mean_absolute_error, median_absolute_error
from pandas.tseries.offsets import *
import datetime as dt
from calendar import monthrange
import datetime
from pandas.tseries.offsets import MonthEnd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from apiclient import discovery
import tensorflow as tf
from tensorflow.contrib import learn
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import sys
import json
# import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

########################################################################
##      LOAD THE DATA
########################################################################
df = pd.read_csv("/home/brett/Downloads/SmallDeptList.csv")
print(df.shape)

## isolate single dept
df = df[df.DEPARTMENT_ID == 102011]

idx = pd.date_range(df.VISIT_DTM.min(), df.VISIT_DTM.max())

## aggregation by date
df = df.groupby('VISIT_DTM').count().reset_index()

df.loc[:, 'VISIT_DTM'] = pd.to_datetime(df.loc[:, 'VISIT_DTM'])
df = df.set_index('VISIT_DTM')

df = df.reindex(idx, fill_value=0)
df = df.reset_index()

df.columns = ['ds', 'y']

## ADD COLUMNS
df.loc[:, 'weekend_flag'] = np.where(
        df.ds.dt.weekday_name.isin(['Saturday', 'Sunday']),
    1, 0)

df['year'], df['month'], df['day'] = df['ds'].dt.year, df['ds'].dt.month, df['ds'].dt.day
df['eom'] = pd.to_datetime(df['ds']).dt.days_in_month

df.loc[:, 'eom_flag'] = np.where(df.eom - df.day <= 2, 1, 0)

cal = calendar()
holidays = cal.holidays(start=df.ds.min(), end=df.ds.max())

df.loc[:, 'holiday_flag'] = np.where(
    df.ds.isin(holidays), 1, 0
)

print(df.shape)

FEATURES = [
    "weekend_flag",
    "eom_flag",
    "holiday_flag"
]

ml = discovery.build('ml', 'v1')
projectID = 'projects/{}'.format('infusionsft-looker-poc')

for i in range(1,11):
    i_str = str(i)
    df.loc[:, str('y_l' + i_str)] = df.y.shift(i).fillna(0)

df = df[df.ds > '2015-01-12']

x = df.iloc[:, np.r_[2,7:19]]
y = df.loc[:, 'y']

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=.90,
    test_size=.10,
    shuffle=False
)

x_train = x_train.values
x_test = x_test.values
y_train = y_train.values
y_test = y_test.values

# reshape input to be 3D [samples, timesteps, features]
x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

model = Sequential()
model.add(LSTM(128, input_shape=(x_train.shape[1], x_train.shape[2])))
# model.add(Dense(1, activation='sigmoid'))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# # model.compile(loss='mse', optimizer='rmsprop')
# model.compile(
#     loss='binary_crossentropy',
#     optimizer='rmsprop',
#     metrics=['accuracy']
# )

# fit network
model.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=72,
    validation_data=(x_test, y_test),
    verbose=2,
    shuffle=False
)

# make a prediction
yhat = model.predict(x_test)
x_test = x_test.reshape((x_test.shape[0], x_test.shape[2]))
yhat = pd.DataFrame(yhat)
y_test = pd.DataFrame(y_test)
y_test = pd.concat([y_test.reset_index(drop=True), yhat], axis=1)
y_test.columns = ['y', 'pred']
y_test.pred = np.where(y_test.pred < 0, 0, y_test.pred)

# mean square error
mse = mean_squared_error(
    y_true=y_test.loc[:, "y"],
    y_pred=y_test.loc[:, "pred"]
)
# root mean square error
rmse = np.sqrt(mse)

# mean absolute error
mae = mean_absolute_error(
    y_true=y_test.loc[:, "y"],
    y_pred=y_test.loc[:, "pred"]
)

mdae = median_absolute_error(
    y_true=y_test.loc[:, "y"],
    y_pred=y_test.loc[:, "pred"]
)

exp_var = explained_variance_score(
    y_true=y_test.loc[:, "y"],
    y_pred=y_test.loc[:, "pred"]
)

print("RMSE: {0}".format(rmse))
print("MSE: {0}".format(mse))
print("MAE: {0}".format(mae))
print("MED. AE: {0}".format(mdae))
print("EXP VAR: {0}".format(exp_var))




## CURRENT BEST
#model = Sequential()
#model.add(LSTM(128, input_shape=(x_train.shape[1], x_train.shape[2])))
#model.add(Dense(1))
#model.compile(loss='mae', optimizer='adam')

# RMSE: 42.36566305871971
# MSE: 1794.849406404968
# MAE: 31.125658281352543
# MED. AE: 26.314239501953125
# EXP VAR: 0.6927468755096203



# plot history
pyplot.plot(model.history['loss'], label='train')
pyplot.plot(model.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()








