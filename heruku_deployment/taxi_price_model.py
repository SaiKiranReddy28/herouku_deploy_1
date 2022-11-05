import numpy as np # linear algebra
import pandas as pd
import random
import math
from datetime import date
import calendar
import matplotlib.pyplot as plt
import pickle

df=pd.read_csv("C:/Users/SAI/Downloads/Users (1).csv")

#drop rows if the passange_count=0
index_names=df.loc[df["passenger_count"]==0].index
df.drop(index_names,inplace=True)

#drop rows if fare_amount is 0 or negative
index_names=df.loc[df["fare_amount"]<=0].index
df.drop(index_names,inplace=True)

df["day"]=pd.to_datetime(df["pickup_datetime"]).dt.day_name()
df["time"]=pd.to_datetime(df["pickup_datetime"]).apply(lambda x:x.hour)


dict={"Sunday":0,"Monday":1,"Tuesday":2,"Wednesday":3,"Thursday":4,"Friday":5,"Saturday":6}
df["day"].replace(dict,inplace=True)

def haversine(lon1, lat1, lon2, lat2, to_radians=True, earth_radius=6371):
    if to_radians:
        lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

    a = np.sin((lat2-lat1)/2.0)**2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin((lon2-lon1)/2.0)**2
    val=earth_radius * 2 * np.arcsin(np.sqrt(a))
    #print(val)
    return val

#calculate_distance of travel
val=[]
for a,b,c,d in zip(df["pickup_longitude"],df["pickup_latitude"],df["dropoff_longitude"],df["dropoff_latitude"]):
    val.append(haversine(a,b,c,d))
df["distance"]=np.array(val)

from sklearn.preprocessing import StandardScaler
sd=StandardScaler()
col_to_scale=["distance"]
sd.fit(df[col_to_scale])
df[col_to_scale]=sd.transform(df[col_to_scale])

target=df["fare_amount"]
drop_cols=["key","pickup_datetime","fare_amount"]
df1=df.drop(drop_cols,axis=1)

target=np.array(target,dtype="float64")

from sklearn.model_selection import train_test_split
X_train,X_valid,y_train,y_valid=train_test_split(df1,target,test_size=0.2,random_state=42,shuffle=True)

from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor(n_estimators=300,min_samples_split=100,max_depth=10,max_leaf_nodes=10,bootstrap=True,n_jobs=-1,oob_score=True)
rfr.fit(X_train,y_train)

import math
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
y_pred_rfr=rfr.predict(X_valid)
print("MAE:",mean_absolute_error(y_pred_rfr,y_valid))
print("MSE:",mean_squared_error(y_pred_rfr,y_valid))
print("RMSE:",math.sqrt(mean_squared_error(y_pred_rfr,y_valid)))
print("R2_Score:",r2_score(y_pred_rfr,y_valid))

pickle.dump(rfr,open('model.sav','wb'))
model=pickle.load(open('model.sav',"rb"))
print(model.score(X_valid,y_valid))
print(model.predict([[-73.990930,40.765603,-73.982563,40.770270,1,1,10,-0.050647]]))


