import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from data_collection import raw_housing_Beijing
from sklearn import preprocessing
from data_collection import stations


raw_housing_Beijing = raw_housing_Beijing.drop(["id", "Cid", "followers","tradeTime","fiveYearsProperty"],axis = 1) 

housing_Beijing = raw_housing_Beijing.fillna(axis=0,method='backfill') # 12, "ConstructionTime" has one null entry!
housing_Beijing = housing_Beijing.dropna(subset=["constructionTime"])


housing_Beijing_selected_zone = housing_Beijing[housing_Beijing["Lng"] < 120.0]
housing_Beijing_selected_zone = housing_Beijing[housing_Beijing["Lng"] >= 116.4]

housing_Beijing_selected_zone = housing_Beijing[housing_Beijing["Lat"] < 40.0]
housing_Beijing_selected_zone = housing_Beijing[housing_Beijing["Lat"] >= 39.930]


#Split data into test and validation sets
x_pd = housing_Beijing_selected_zone.drop([#"totalPrice","price",
	"livingRoom","drawingRoom","kitchen",
	"bathRoom","buildingType",
	"buildingStructure","ladderRatio","elevator",
	"subway","district","communityAverage"
	],axis=1)
print(x_pd)

X = x_pd.values #.iloc[:50000,:]
y = housing_Beijing_selected_zone.iloc[:, 4].values#50000

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.25, random_state = 0)


#Since for the most time, only opening date and locations are needed
stations_locations = stations.iloc[:, 4:7]

