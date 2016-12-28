# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 17:22:22 2016

@author: mckenka
"""


from sklearn.ensemble.forest import RandomForestRegressor
import pandas as pd
import os
import numpy as np


os.chdir("C:\\Users\\mckenka\\Documents\\Cycle Competition")

train=pd.read_excel(".\\Use\\train_cleaned.xlsx")
test=pd.read_excel(".\\Use\\test_cleaned.xlsx")
new_test=test.head(n=0)
new_test["prediction"]=new_test["cadence"]

for i in range(1,16):
    print(i)
    
    ##predict cadence for test dataset
    train_i=train.loc[train["rider_id"]==i]
    X=np.array(train_i[["avg_gradient","max_gradient","distance","highest_point","lowest_point",
                "measured_time","moving_time","avg_heart_rate","max_heart_rate","power","speed"]])
    y=np.array(train_i[["cadence"]])
    y=np.array(y).astype(float)
    rf_cadence = RandomForestRegressor(n_estimators=500,criterion='mse',max_features='sqrt')
    rf_cadence.fit(X,np.ravel(y))
    
    ##get the test data for prediction
    test_i=test.loc[test["rider_id"]==i]
    test_i=test_i.loc[test_i["cadence"].isnull()]
    X_test=np.array(test_i[["avg_gradient","max_gradient","distance","highest_point","lowest_point",
                "measured_time","moving_time","avg_heart_rate","max_heart_rate","power","speed"]])
    test_i["prediction"]=pd.Series(rf_cadence.predict(X_test)).values

    new_test=new_test.append(test_i,ignore_index=True)
    
    ##predict power for test dataset
    X=np.array(train_i[["avg_gradient","max_gradient","distance","highest_point","lowest_point",
                "measured_time","moving_time","avg_heart_rate","max_heart_rate","cadence","speed"]])
    y=np.array(train_i[["power"]])
    y=np.array(y).astype(float)
    rf_power = RandomForestRegressor(n_estimators=500,criterion='mse',max_features='sqrt')
    rf_power.fit(X,np.ravel(y))
    
    ##get the test data for prediction
    test_i=test.loc[test["rider_id"]==i]
    test_i=test_i.loc[test_i["power"].isnull()]
    X_test=np.array(test_i[["avg_gradient","max_gradient","distance","highest_point","lowest_point",
                "measured_time","moving_time","avg_heart_rate","max_heart_rate","cadence","speed"]])
    test_i["prediction"]=pd.Series(rf_power.predict(X_test)).values

    new_test=new_test.append(test_i,ignore_index=True)
    
    ##predict speed for test dataset - no model
    test_i=test.loc[test["rider_id"]==i]
    test_i=test_i.loc[test_i["speed"].isnull()]
    test_i["prediction"]=test_i["distance"]/test_i["measured_time"]
    new_test=new_test.append(test_i,ignore_index=True)

new_test=new_test.sort(columns="id")
new_test.to_excel(".\\Use\\submission_v1.xlsx",index=False)    

new_test['Id;Prediction']=new_test.apply(lambda x:'%s;%s' % (x['id'],x['prediction']),axis=1)
new_test['Id;Prediction'].to_csv(".\\Output\\submission_v1.csv",index=False,header=True)    
    