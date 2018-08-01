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

FEATURES = [
    "weekend_flag",
    "eom_flag"
]
########################################################################
##      FIND BEST PARAMETERS
########################################################################

# separate features from target
x = df.loc[:, FEATURES].copy()
y = df.loc[:, 'y'].copy()


# create training and testing splits
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=.60,
    test_size=.40,
    shuffle=False
)

# get the indices
train_idx = x_train.index
test_idx = x_test.index

scores = collections.OrderedDict()

# placeholder values
best_order = (-1, -1, -1, -1, -1, -1, -1)
best_rmse = np.inf

# get every permutation of AR-I-MA values to try
AR_values = list(range(1, 8))
I_values= list(range(1, 8))
MA_values = list(range(1, 8))
# number of periods in a "season"; periodicity
# P_values = list(range(1, 3))
# D_values = list(range(1, 8))
# Q_values = list(range(1, 3))
# S_values = list(range(1, 30))
SARIMAX_values = [AR_values, I_values, MA_values]

SARIMAX_permutations = list(itertools.product(*SARIMAX_values))


for permutation in SARIMAX_permutations:
    print("Running SARIMAX with order {0}".format(permutation))
    AR=permutation[0]
    # we've manually handled differencing where applicable so no
    # Integration term is needed
    I=permutation[1]
    MA=permutation[2]
    # # excluding since we don't have enough samples for daily periodicity
    # P=permutation[3]
    # D=permutation[4]
    # Q=permutation[5]
    # s=permutation[6]

    sarimax = sm.tsa.SARIMAX(
        df.loc[train_idx, 'y'].astype(np.float64),
        exog=df.loc[train_idx, FEATURES],
        order=(AR,I,MA),
        # seasonal_order=(P,D,Q,s),
        simple_differencing=False)

    try:
        fitted_sarimax = sarimax.fit(
            maxiter=1000,
            cov_type="robust"
        )

        # check for convergence
        converged = fitted_sarimax.mle_retvals["converged"]

        # if the model doesn't converge, we'll discard the order as a
        # possibility
        if converged is False:
            print("Model failed to converge")
            raise Exception

        # make in-sample predictions
        is_model = sm.tsa.SARIMAX(
            df.loc[:, 'y'].astype(np.float64),
            exog=df.loc[:, FEATURES],
            order=(AR, I, MA),
            # seasonal_order=(P,D,Q,s),
            simple_differencing=False)

        is_results = is_model.filter(fitted_sarimax.params)

        # make in-sample predictions
        pred = is_results.predict()

        train_aic = fitted_sarimax.aic
        test_aic = is_results.aic

        # some of these could fail depending on the output
        # statsmodels produces...man this library is high-maintenance...

        mse = mean_squared_error(y_true=df.loc[test_idx, 'y'],
                                 y_pred=pred[test_idx])

        # root mean square error
        rmse = np.sqrt(mse)

        # mean absolute error
        mae = mean_absolute_error(y_true=df.loc[test_idx, 'y'],
                                  y_pred=pred[test_idx])

        mdae = median_absolute_error(y_true=df.loc[test_idx, 'y'],
                                     y_pred=pred[test_idx])

        exp_var = explained_variance_score(
            y_true=df.loc[test_idx, 'y'],
            y_pred=pred[test_idx])


        scores[str(permutation)] = collections.OrderedDict()
        scores[str(permutation)]["train_aic"] = train_aic
        scores[str(permutation)]["test_aic"] = test_aic
        scores[str(permutation)]["mse"] = mse
        scores[str(permutation)]["rmse"] = rmse
        scores[str(permutation)]["mae"] = mae
        scores[str(permutation)]["mdae"] = mdae
        scores[str(permutation)]["explained_variance"] = exp_var
        scores[str(permutation)]["model"] = fitted_sarimax
        scores[str(permutation)]["test_data"] = test_idx
        scores[str(permutation)]["predictions"] = pred[test_idx]
        scores[str(permutation)]["converged"] = converged

        if rmse <= best_rmse:
            best_rmse = rmse
            best_order = permutation

    except:
        continue


print("Best order for SARIMAX: {0}".format(str(best_order)))
print("RMSE: {0}".format(best_rmse))


# best model params
AR = 4
I = 1
MA = 5
# s =

train_df = df[df.ds <= '2017-10-01']
test_df = df[(df.ds > '2017-10-01') & (df.ds <= '2017-10-11')]

sarimax = sm.tsa.statespace.SARIMAX(
    train_df.loc[:, 'y'].astype(np.float64),
    exog=train_df.loc[:, FEATURES],
    order=(AR, I, MA),
    # seasonal_order=(P,D,Q,s),
    simple_differencing=False,
    enforce_stationarity=False,
    enforce_invertibility=False
)

fitted_sarimax = sarimax.fit(
    maxiter=1000,
    cov_type="robust",
)

# print the results
print(fitted_sarimax.summary())

# preds = fitted_sarimax.get_forecast(
#     steps=10,
#     exog=test_df.loc[:, FEATURES]
# )
# oos_model = sm.tsa.SARIMAX(
#     test_df.loc[:, 'y'],
#     exog=test_df.loc[:, FEATURES],
#     order=(AR, I, MA),
#     # seasonal_order=(P,D,Q,s),
#     simple_differencing=False
# )

# oos_results = oos_model.filter(fitted_sarimax.params)
# # make in-sample predictions
#
# preds = oos_results.predict()
# preds = np.round(preds, decimals=0)

preds = fitted_sarimax.forecast(
    test_df.shape[0],
    exog=test_df.loc[:, FEATURES]
)

preds = pd.DataFrame(preds)
preds.columns = ['pred']
test_df = pd.DataFrame(test_df)

oos_df = pd.concat(
    [
        test_df.reset_index(drop=False),
        preds.reset_index(drop=False)
    ],
    axis=1
)

new_names = oos_df.columns.values
new_names[10] = "pred"
oos_df.columns = new_names

# mean square error
mse = mean_squared_error(
    y_true=oos_df.loc[:, "y"],
    y_pred=oos_df.loc[:, "pred"]
)
# root mean square error
rmse = np.sqrt(mse)

# mean absolute error
mae = mean_absolute_error(
    y_true=oos_df.loc[:, "y"],
    y_pred=oos_df.loc[:, "pred"]
)

mdae = median_absolute_error(
    y_true=oos_df.loc[:, "y"],
    y_pred=oos_df.loc[:, "pred"]
)

exp_var = explained_variance_score(
    y_true=oos_df.loc[:, "y"],
    y_pred=oos_df.loc[:, "pred"]
)

print("RMSE: {0}".format(rmse))
print("MSE: {0}".format(mse))
print("MAE: {0}".format(mae))
print("MED. AE: {0}".format(mdae))
print("EXP VAR: {0}".format(exp_var))

# MSE: 22.419598529469116
# MSE: 502.6383982225737
# MAE: 19.648158607975105
# MED. AE: 21.156291948027423
# EXP VAR: 0.7955351209114689


