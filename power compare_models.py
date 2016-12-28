# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 12:28:25 2016

@author: mckenka
"""

##compare prediction accuracy of RF, Linear Model, and SVR - look at MSE for power, speed and cadence

from sklearn import linear_model
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm.classes import SVR
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
import pandas as pd
import os
import numpy as np
import math


os.chdir("C:\\Users\\mckenka\\Documents\\Cycle Competition")

train=pd.read_excel(".\\Use\\train_cleaned.xlsx")

##compare prediction accuracy of RF, Linear Model, and SVR - look at MSE for power

##do this for player 1,...,15 
results=pd.DataFrame({'model': [],"rider":[],"rmse_validated":[]})
for i in range(1,16):
    print(i)
    train_i=train.loc[train["rider_id"]==i]
    X=np.array(train_i[["avg_gradient","max_gradient","distance","highest_point","lowest_point",
                "measured_time","moving_time","avg_heart_rate","max_heart_rate","speed","cadence"]])
    y=np.array(train_i[["power"]])
    y=np.array(y).astype(float)
    X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.2, random_state=0)
    
    
    ##linear regression
    regr = linear_model.LinearRegression()
    regr.fit(X_train, np.ravel(y_train))
    rmse=math.sqrt(mean_squared_error(y_validate, regr.predict(X_validate)))
    results=results.append(pd.DataFrame([["Linear Regression",i,rmse]],columns=["model","rider","rmse_validated"]),ignore_index=True)
    
    ##linear regression - LassoLars
    regr = linear_model.LassoLars(alpha=.1)
    regr.fit(X_train, np.ravel(y_train))
    rmse=math.sqrt(mean_squared_error(y_validate, regr.predict(X_validate)))
    results=results.append(pd.DataFrame([["Linear Regression Lasso Lars",i,rmse]],columns=["model","rider","rmse_validated"]),ignore_index=True)
    
    ##random forest
    regr_rf = RandomForestRegressor(n_estimators=100,criterion='mse',max_features='sqrt')
    regr_rf.fit(X_train,np.ravel(y_train))
    rmse = math.sqrt(mean_squared_error(y_validate, regr_rf.predict(X_validate)))
    results=results.append(pd.DataFrame([["Random Forest",i,rmse]],columns=["model","rider","rmse_validated"]),ignore_index=True)
    
    ##do gradient boosting regression
    gradient = GradientBoostingRegressor()
    gradient.fit(X_train,np.ravel(y_train))
    rmse = math.sqrt(mean_squared_error(y_validate, gradient.predict(X_validate)))
    results=results.append(pd.DataFrame([["Gradient Boosting",i,rmse]],columns=["model","rider","rmse_validated"]),ignore_index=True)
    
    ##SVR
    regr_svr = SVR()
    regr_svr.fit(X_train,np.ravel(y_train))
    rmse = math.sqrt(mean_squared_error(y_validate, regr_svr.predict(X_validate)))
    results=results.append(pd.DataFrame([["SVR",i,rmse]],columns=["model","rider","rmse_validated"]),ignore_index=True)

results.to_excel(".\\Use\\power_compare_base_models.xlsx",index=False)