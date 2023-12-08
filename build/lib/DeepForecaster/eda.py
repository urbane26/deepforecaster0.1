#import core pkgs
import pandas as pd
import numpy as np
from PIL import Image

#utility pkgs
import pathlib
from pathlib import Path
import glob
import time
import shutil

#core plotting pkgs
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import plotly.express as px
import seaborn as sns
sns.set_theme(context='notebook', style='darkgrid', palette='deep')

#core metric pkgs
from sklearn import metrics
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.inspection import permutation_importance

#core forecasting and analysis pkgs
import torch
import shap
import xgboost as xgb
from xgboost import XGBRegressor
from tsmoothie.smoother import LowessSmoother
from statsmodels.tsa.stattools import adfuller
import antropy as ant

#suppporting functions
def forecastability(colname):
    ApEn = ant.app_entropy(colname)
    return ApEn

def corranalysis(dataset : pd.DataFrame):
    dataset_c = dataset.select_dtypes(include=['number'])
    corr = dataset_c.corr()
    return corr
#main functions for EDA

def shapanalysis(dataset: pd.DataFrame):
    shapes = dataset.shape
    featureset = dataset.drop(['ds','y','unique_id'],axis =1).columns.to_list()
    k = int(.80 *len(dataset))
    y= dataset['y']
    X = dataset[featureset]
    y_train =  y.iloc[0:k]
    y_test = y.iloc[k:shapes[0]]    
    X_train = X.iloc[0:k,:]
    X_test = X.iloc[k:shapes[0],:]

    xgb = XGBRegressor(objective= 'reg:squarederror', n_estimators=1000,booster = 'gbtree')
    model = xgb.fit(X_train, y_train,eval_set=[(X_train, y_train), (X_test, y_test)],early_stopping_rounds=50,verbose=False)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, features=X, feature_names=X.columns, plot_type='bar')
    g = plt.gcf()
    return g



def stationaritychk(colname : pd.Series):
    adf_test = adfuller(colname)
    ADF_Statistic = adf_test[0]
    p_value = adf_test[1]
    if p_value < 0.05:
        str1 = 'This time series is stationary'
    else:
        str1 = 'This time series is not stationary'
    return str1

def DemandClassification(dataset : pd.DataFrame):
    Xavg = np.mean(dataset['y'] )
    Xsd = np.std(dataset['y'])
    CVsq = (Xsd/Xavg)**2
    X1 = (dataset['y'] != 0).sum()
    AD_I = X1/len(dataset['y'] )

    if AD_I < 1.32 and CVsq < 0.49:
        str1 = 'Smooth Demand. Low error margin expected'
    elif AD_I >= 1.32 and CVsq < 0.49:
        str1 = 'Intermittend Demand. Higher error margin expected. Try SESOpt algorithm'
    elif AD_I < 1.32 and CVsq >= 0.49:
        str1 = "erratic Demand. Very high error margin expected"
    else:
        str1 = 'Lumpy Demand. No reasonable accuracy is expected'
    return str1
   
    


