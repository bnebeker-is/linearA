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

#### CONFIGS ###########################################################
########################################################################
DATA_PATH="/home/jason/code/python/marketing_mix_model/data" \
          "/daily/training_data.csv.gz"

SAVE_PATH="/home/jason/code/python/marketing_mix_model" \
          "/final_model_assets/"

DATE_FEATURE=["date_dt"]
FEATURES=[
    "weekend_flag",
    "eom_flag"
]





########################################################################
##      LOAD THE DATA
########################################################################
df = pd.read_csv("/home/brett/Downloads/SmallDeptList.csv")
print(df.shape)

## isolate single dept
df = df[df.DEPARTMENT_ID == 102011]

## aggregation by date
df = df.groupby('VISIT_DTM').count().reset_index()

df.loc[:, 'VISIT_DTM'] = pd.to_datetime(df.loc[:, 'VISIT_DTM'])

df = df.reindex(idx, fill_value=0)

df.loc[:, 'weekend_flag'] = np.where(
    df.VISIT_DTM.dt.weekday_name.isin(['Saturday', 'Sunday']),
        1, 0)


########################################################################
##      FIND BEST PARAMETERS
########################################################################









# best model params
#AR=
#I=
#MA=
# s=

sarimax = sm.tsa.statespace.SARIMAX(
    df.loc[:, TARGET].astype(np.float64),
    exog=df.loc[:, FEATURES],
    order=(AR,I,MA),
    # seasonal_order=(P,D,Q,s),
    simple_differencing=False,
enforce_stationarity=False,
enforce_invertibility=False)

fitted_sarimax = sarimax.fit(
    maxiter=1000,
    cov_type="robust",
)

# doesn't work as of v0.8; you'd have to add the commit that contains
# the code to allow this as it wasn't included in the 0.8 release
# despite being merged prior; has to do with some upgrades the Cython
# see: https://github.com/statsmodels/statsmodels/issues/3892
# fitted_sarimax.save(SAVE_PATH + "fitted_sarimax.ml")