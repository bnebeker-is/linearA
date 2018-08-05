## CURRENT BEST RNN

import collections
import itertools
import numpy as np
import pandas as pd
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
from sklearn.preprocessing import MinMaxScaler

import sys
import json
# import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


########################################################################
##      LOAD THE DATA
########################################################################
df = pd.read_csv("/home/brett/Downloads/SmallDeptList.csv")
print(df.shape)

# df_g = df.groupby(['VISIT_DTM', 'DEPARTMENT_ID']).size().reset_index(name='counts')

## isolate single dept
## TEST THESE DEPTS
df = df[df.DEPARTMENT_ID == 102011]
df = df[df.DEPARTMENT_ID == 102022]
df = df[df.DEPARTMENT_ID == 102052]
df = df[df.DEPARTMENT_ID == 104067]
df = df[df.DEPARTMENT_ID == 106022]

idx = pd.date_range(df.VISIT_DTM.min(), df.VISIT_DTM.max())

# df.groupby(['VISIT_DTM', 'DEPARTMENT_ID']).size()

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

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(x)
scaled = pd.DataFrame(scaled)

x_train, x_test, y_train, y_test = train_test_split(
    scaled,
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
# model.add(LSTM(1, batch_input_shape=(2, x_train.shape[1], x_train.shape[2]), stateful=True))
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
    epochs=1000,
    batch_size=50,
    validation_data=(x_test, y_test),
    verbose=2,
    shuffle=False
)

# make a prediction
yhat = model.predict(x_test)
x_test_rs = x_test.reshape((x_test.shape[0], x_test.shape[2]))
yhat = pd.DataFrame(yhat)
y_test_rs = pd.DataFrame(y_test)
y_test_rs = pd.concat([y_test_rs.reset_index(drop=True), yhat], axis=1)
y_test_rs.columns = ['y', 'pred']
y_test_rs.pred = np.where(y_test_rs.pred < 0, 0, y_test_rs.pred)

# train pred
yhat_train = model.predict(x_train)
x_train_rs = x_train.reshape((x_train.shape[0], x_train.shape[2]))
yhat_train = pd.DataFrame(yhat_train)
y_train_rs = pd.DataFrame(y_train)
y_train_rs = pd.concat([y_train_rs.reset_index(drop=True), yhat_train], axis=1)
y_train_rs.columns = ['y', 'pred']
y_train_rs.pred = np.where(y_train_rs.pred < 0, 0, y_train_rs.pred)

# mean square error
mse = mean_squared_error(
    y_true=y_test_rs.loc[:, "y"],
    y_pred=y_test_rs.loc[:, "pred"]
)
# root mean square error
rmse = np.sqrt(mse)

# mean absolute error
mae = mean_absolute_error(
    y_true=y_test_rs.loc[:, "y"],
    y_pred=y_test_rs.loc[:, "pred"]
)

mdae = median_absolute_error(
    y_true=y_test_rs.loc[:, "y"],
    y_pred=y_test_rs.loc[:, "pred"]
)

exp_var = explained_variance_score(
    y_true=y_test_rs.loc[:, "y"],
    y_pred=y_test_rs.loc[:, "pred"]
)

print("RMSE: {0}".format(rmse))
print("MSE: {0}".format(mse))
print("MAE: {0}".format(mae))
print("MED. AE: {0}".format(mdae))
print("EXP VAR: {0}".format(exp_var))

# mean square error
mse_t = mean_squared_error(
    y_true=y_train_rs.loc[:, "y"],
    y_pred=y_train_rs.loc[:, "pred"]
)
# root mean square error
rmse_t = np.sqrt(mse_t)

# mean absolute error
mae_t = mean_absolute_error(
    y_true=y_train_rs.loc[:, "y"],
    y_pred=y_train_rs.loc[:, "pred"]
)

mdae_t = median_absolute_error(
    y_true=y_train_rs.loc[:, "y"],
    y_pred=y_train_rs.loc[:, "pred"]
)

exp_var_t = explained_variance_score(
    y_true=y_train_rs.loc[:, "y"],
    y_pred=y_train_rs.loc[:, "pred"]
)

## calc differences
rmse_diff = rmse_t - rmse
mse_diff = mse_t - mse
mae_diff = mae_t - mae
mdae_diff = mdae_t - mdae
exp_var_diff = exp_var_t - exp_var

## print results
print("")
print("TRAIN SET RESULTS:")
print("RMSE: {0}".format(rmse_t))
print("MSE: {0}".format(mse_t))
print("MAE: {0}".format(mae_t))
print("MED. AE: {0}".format(mdae_t))
print("EXP VAR: {0}".format(exp_var_t))
print("")
print("RMSE DIFF {0}".format(rmse_diff))
print("EXP VAR DIFF {0}".format(exp_var_diff))



y_test_rs.to_csv("/home/brett/rnn2.csv")

