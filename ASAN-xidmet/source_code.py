import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import gc
import seaborn as sns
import xgboost as xgb
from xgboost import XGBClassifier
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, classification_report, roc_auc_score
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.graph_objs import Line
import plotly.express as px
import catboost
import seaborn as sn
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 300)
pd.set_option('display.max_columns', 300)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
# plot sizes
plt.rcParams["figure.figsize"] = [15, 10]


# --- Read Data ---
q_df = pd.read_csv("/kaggle/input/ai4digigov2021/queue_dataset_train.csv")
q_df_test = pd.read_csv("/kaggle/input/ai4digigov2021/queue_dataset_test.csv")
q_df_sub = pd.read_csv("/kaggle/input/ai4digigov2021/baseline_submission.csv")

q_df.info()

# --- Convert date into pandas date ---
q_df["date_converted"] = pd.to_datetime(q_df["date"], format='%Y-%m-%d')
q_df_test["date_converted"] = pd.to_datetime(q_df_test["date"], format='%Y-%m-%d')

# --- Holiday Extractor
sorted_dates_train = sorted(q_df_test["date_converted"].dt.date.unique().tolist())
date_set = set(sorted_dates_train[0] + timedelta(x) for x in range((sorted_dates_train[-1] - sorted_dates_train[0]).days))
train_dataset_days_with_no_service = sorted(date_set - set(sorted_dates_train))
train_dataset_days_with_no_service

# --- Print NA ---
q_df.isna().sum()

# --- Impute NA values ---

# impute ages
q_df["customer_age_appl"].fillna(q_df["customer_age_appl"].mode().iloc[0], inplace=True)

# impute ticket hours
q_df["ticket_taking_time_hour"] = q_df['time_start_process'].str[:2] 
q_df["ticket_taking_time_hour"] = q_df["ticket_taking_time_hour"].astype(str)
q_df['ticket_taking_time_hour'].fillna(q_df['ticket_taking_time_hour'].mode().iloc[0], inplace= True)
q_df["ticket_taking_time_hour"] = q_df["ticket_taking_time_hour"].replace("nan", q_df['ticket_taking_time_hour'].mode().iloc[0])

# --- Extract Separate dates from date ---

# extract month 
q_df["approach_month"] = q_df["date_converted"].dt.month.astype(str)

# extract weekdays
week_days={0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}
q_df['weekday'] = q_df['date_converted'].dt.dayofweek.map(week_days)

# --- Drop Columns ---

# - CORRELATION of [operator_count , previous_customer_count] 
#df = q_df[["operator_count","previous_customer_count"]]
#corrMatrix = df.corr()
#sn.heatmap(corrMatrix, annot=True)
#plt.show()
#RESULT: 0.43 corr --> DROP NONE

# - Already used, so UNNECESSARY columns dropped
# Note: dont drop service_name, instead drop service_name_2, because there are more varience in service_name
q_df.drop(["id", "date", "time_start_process", "date_converted"], axis=1, inplace=True) 

# ------------ Theirs Feature Selection for X -----------
# [["operator_count", "customer_gender", 'customer_age_appl',
# 'customer_city', "approach_month", "ticket_taking_time_hour"]]

# ------------ Our Feature Selection for X ------------
# dataset = q_df[["branch_name", "customer_gender", "customer_age_appl", "customer_city",
#                 "service_name_organization", "service_name", "service_name_2",
#                 "operator_count", "previous_customer_count",
#                "approach_month","ticket_taking_time_hour", "weekday"]]


# ================================================== Model Selection & Hyperparameter Tuning ============================================================
import sys
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
sys.path.insert(1, '../input/paramsearch/')

from paramsearch import paramsearch
from itertools import product,chain
from sklearn import metrics

## Grid Search + Cross Validation for Hyperparameter Tuning
# params = {'depth':[2,3,5],                 #'depth':[2,3,4,5],
#           'iterations':[250,500,1000],              #'iterations':[500,1000],
#           'learning_rate':[0.01,0.1,0.3],   #'learning_rate':[0.05, 0.1, 0.3], 
#           'l2_leaf_reg':[3,5],              #'l2_leaf_reg':[1,3,5],
#           'border_count':128,
#           'thread_count':4,
#           'task_type':"GPU",
#           'loss_function':'Logloss',
#           'eval_metric':"AUC",
#           'auto_class_weights':"Balanced",
#           'simple_ctr':None,
#           'verbose':True}

# def crossvaltest(params,splits=None):
#     skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
#     skf_train_dev = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
#     service = q_df['service_canceled']
#     res = []
#     for train_index, test_index in skf.split(q_df, service):
#         train = q_df.loc[train_index, :]
#         test = q_df.loc[test_index, :]
        
#         y_train = train['service_canceled'] 
#         x_train = train.drop(["service_canceled"], axis=1)
        
#         y_test = test['service_canceled']
#         x_test = test.drop(["service_canceled"], axis=1)

#         categorical_features_indices = [0,1,2,3,4,5,6,9,10,11]
#         train_pool = catboost.Pool(x_train, y_train, cat_features=categorical_features_indices)

#         model = catboost.CatBoostClassifier(**params)
#         model.fit(train_pool)
        
#         y_pred = model.predict_proba(x_test)[:, 1]
#         auc = roc_auc_score(y_test, y_pred)
#         res.append(auc)
#     return np.mean(res)

# def catboost_param_tune(params):
#     ps = paramsearch(params)
#     for prms in chain(ps.grid_search(['l2_leaf_reg']),
#                       ps.grid_search(['iterations','learning_rate']),
#                       ps.grid_search(['depth'])):
#         res = crossvaltest(prms,splits=4)
#         ps.register_result(res,prms)
#         print(f"{res},{prms},best:,{ps.bestscore()},{ps.bestparam()}")
#     return ps.bestparam()

# bestparams = catboost_param_tune(params)

# bestparams
# clf = catboost.CatBoostClassifier(**bestparams)
# categorical_features_indices = [0,1,2,3,4,5,6,9,10,11]

# labels = q_df['service_canceled']
# dataset = q_df.drop(["service_canceled"], axis=1)
    
# train_pool = catboost.Pool(dataset, labels, cat_features=categorical_features_indices)
# clf.fit(train_pool)

bestparams = {'depth': 4,
 'iterations': 400,
 'learning_rate': 0.1,
 'l2_leaf_reg': 7,
 'border_count': 128,
 'thread_count': 4,
 'task_type': 'GPU',
 'loss_function': 'Logloss',
 'eval_metric': 'AUC',
 'auto_class_weights': 'Balanced',
 'simple_ctr': None,
 'verbose': True}


clf_1 = catboost.CatBoostClassifier(**bestparams)
categorical_features_indices = [0,1,2,3,4,5,6,9,10,11]

labels = q_df['service_canceled']
dataset = q_df.drop(["service_canceled"], axis=1)
    
train_pool = catboost.Pool(dataset, labels, cat_features=categorical_features_indices)
clf_1.fit(train_pool)

bestparams = {'depth': 3,
 'iterations': 800,
 'learning_rate': 0.1,
 'l2_leaf_reg': 6,
 'border_count': 128,
 'thread_count': 4,
 'task_type': 'GPU',
 'loss_function': 'Logloss',
 'eval_metric': 'AUC',
 'auto_class_weights': 'Balanced',
 'simple_ctr': None,
 'verbose': True}


clf_2 = catboost.CatBoostClassifier(**bestparams)
categorical_features_indices = [0,1,2,3,4,5,6,9,10,11]

labels = q_df['service_canceled']
dataset = q_df.drop(["service_canceled"], axis=1)
    
train_pool = catboost.Pool(dataset, labels, cat_features=categorical_features_indices)
clf_2.fit(train_pool)

bestparams = {'depth': 5,
 'iterations': 250,
 'learning_rate': 0.1,
 'l2_leaf_reg': 6,
 'border_count': 128,
 'thread_count': 4,
 'task_type': 'GPU',
 'loss_function': 'Logloss',
 'eval_metric': 'AUC',
 'auto_class_weights': 'Balanced',
 'simple_ctr': None,
 'verbose': True}


clf_3 = catboost.CatBoostClassifier(**bestparams)
categorical_features_indices = [0,1,2,3,4,5,6,9,10,11]

labels = q_df['service_canceled']
dataset = q_df.drop(["service_canceled"], axis=1)
    
train_pool = catboost.Pool(dataset, labels, cat_features=categorical_features_indices)
clf_3.fit(train_pool)

bestparams = {'depth': 2,
 'iterations': 1000,
 'learning_rate': 0.05,
 'l2_leaf_reg': 2,
 'border_count': 128,
 'thread_count': 4,
 'task_type': 'GPU',
 'loss_function': 'Logloss',
 'eval_metric': 'AUC',
 'auto_class_weights': 'Balanced',
 'simple_ctr': None,
 'verbose': True}


clf_4 = catboost.CatBoostClassifier(**bestparams)
categorical_features_indices = [0,1,2,3,4,5,6,9,10,11]

labels = q_df['service_canceled']
dataset = q_df.drop(["service_canceled"], axis=1)
    
train_pool = catboost.Pool(dataset, labels, cat_features=categorical_features_indices)
clf_4.fit(train_pool)

# ============================================================ TEST DATA EDA ================================================================================
q_df_test = pd.read_csv("/kaggle/input/ai4digigov2021/queue_dataset_test.csv")
q_df_test["date_converted"] = pd.to_datetime(q_df_test["date"], format='%Y-%m-%d')

# --- Impute NA values ---

# impute ages
q_df_test["customer_age_appl"].fillna(q_df_test["customer_age_appl"].mode().iloc[0], inplace=True)

# impute ticket hours
q_df_test["ticket_taking_time_hour"] = q_df_test['time_start_process'].str[:2] 
q_df_test["ticket_taking_time_hour"] = q_df_test["ticket_taking_time_hour"].astype(str)
q_df_test['ticket_taking_time_hour'].fillna(q_df_test['ticket_taking_time_hour'].mode().iloc[0], inplace= True)
q_df_test["ticket_taking_time_hour"] = q_df_test["ticket_taking_time_hour"].replace("nan", q_df_test['ticket_taking_time_hour'].mode().iloc[0])

# --- Extract Separate dates from date ---

# extract month 
q_df_test["approach_month"] = q_df_test["date_converted"].dt.month.astype(str)

# extract weekdays
week_days={0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}
q_df_test['weekday'] = q_df_test['date_converted'].dt.dayofweek.map(week_days)

# --- Prepare Dataset For Prediction ---
q_df_test.drop(["date", "time_start_process", "date_converted"], axis=1, inplace=True) 
dataset = q_df_test 
dataset = dataset.loc[:, dataset.columns != 'id']

y_pred_test_1 = clf_1.predict_proba(dataset)
y_pred_test_2 = clf_2.predict_proba(dataset)
y_pred_test_3 = clf_3.predict_proba(dataset)
y_pred_test_4 = clf_4.predict_proba(dataset)

y_scores_test = ((y_pred_test_1[:, 1] + y_pred_test_2[:, 1] + y_pred_test_3[:, 1])*3/4 + y_pred_test_4[:, 1]/4)/4

q_df_test['service_canceled'] = y_scores_test
q_df_test[["id", "service_canceled"]].to_csv("submission_4_catboost.csv", index=False)


