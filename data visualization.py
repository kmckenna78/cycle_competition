# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 11:55:40 2016

@author: mckenka
"""

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

os.chdir("C:\\Users\\mckenka\\Documents\\Cycle Competition")

df=pd.read_excel(".\\Use\\train_cleaned.xlsx")

#"rider_id","index","date","time","avg_gradient","max_gradient","distance","highes_point","lowest_point",
 #            "measured_time","moving_time","avg_heart_rate","max_heart_rate","speed","power","cadence"


#do scatterplots of speed with all other variables
def plot_fun(dep_var, ind_var, color):
    fig = sns.jointplot(x=ind_var, y=dep_var, data=df,color=color)
    file_name=".\\Use\\Prelim Data Visualizations\\"+dep_var+"_"+ind_var+".png"
    fig.savefig(file_name)  
    plt.close(fig.fig)

##plot all speed variables
plot_fun("speed","avg_gradient","orange")   
plot_fun("speed","max_gradient","orange")   
plot_fun("speed","distance","orange")  
plot_fun("speed","highest_point","orange")  
plot_fun("speed","lowest_point","orange")  
plot_fun("speed","measured_time","orange")  
plot_fun("speed","moving_time","orange")  
plot_fun("speed","avg_heart_rate","orange")  
plot_fun("speed","max_heart_rate","orange")  
plot_fun("speed","power","orange")  
plot_fun("speed","cadence","orange")  

##plot all power variables
plot_fun("power","avg_gradient","blue")   
plot_fun("power","max_gradient","blue")   
plot_fun("power","distance","blue")  
plot_fun("power","highest_point","blue")  
plot_fun("power","lowest_point","blue")  
plot_fun("power","measured_time","blue")  
plot_fun("power","moving_time","blue")  
plot_fun("power","avg_heart_rate","blue")  
plot_fun("power","max_heart_rate","blue")  
plot_fun("power","speed","blue")  
plot_fun("power","cadence","blue")  

##plot all cadence variables
plot_fun("cadence","avg_gradient","purple")   
plot_fun("cadence","max_gradient","purple")   
plot_fun("cadence","distance","purple")  
plot_fun("cadence","highest_point","purple")  
plot_fun("cadence","lowest_point","purple")  
plot_fun("cadence","measured_time","purple")  
plot_fun("cadence","moving_time","purple")  
plot_fun("cadence","avg_heart_rate","purple")  
plot_fun("cadence","max_heart_rate","purple")  
plot_fun("cadence","speed","purple")  
plot_fun("cadence","power","purple") 