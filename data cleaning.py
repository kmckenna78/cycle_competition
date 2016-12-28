# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 11:49:58 2016

@author: mckenka
"""

import pandas as pd
import os

os.chdir("C:\\Users\\mckenka\\Documents\\Cycle Competition")

df=pd.read_csv(".\\Input Data\\data_train.csv",sep=";")

df=df.ix[:,0].apply(lambda x: pd.Series(x.split(';')))
df.columns=["rider_id","index","date","time","avg_gradient","max_gradient","distance","highest_point","lowest_point",
             "measured_time","moving_time","avg_heart_rate","max_heart_rate","speed","power","cadence"]
             
df.to_excel(".\\Use\\train_cleaned.xlsx",index=False)

df=pd.read_csv(".\\Input Data\\data_test.csv",sep=";")

df=df.ix[:,0].apply(lambda x: pd.Series(x.split(';')))
df.columns=["id","rider_id","index","date","time","avg_gradient","max_gradient","distance","highest_point","lowest_point",
             "measured_time","moving_time","avg_heart_rate","max_heart_rate","speed","power","cadence"]
             
df.to_excel(".\\Use\\test_cleaned.xlsx",index=False)