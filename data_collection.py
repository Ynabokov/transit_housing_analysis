import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import os

print("Program data_collection started!\n")

file_path = "data/housing_price_Beijing.csv"
raw_housing_Beijing = pd.read_csv(file_path,
	encoding = "utf-8", 
	na_values=['nan', 'o', '#NAME?', 'δ֪', '�� 14', '�� 15','�� 16', '�� 6',
	'�� 12','�� 28','�� 11','�� 24','�� 20','�� 22'],
	dtype={0:str,1:float,2:float,3:float, 5:float, 6:float,
	7:float,8:float,9:float,10:float,11:float,12:float,
	13:float,14:float,15:float,16:float,17:float,18:float,19:float,
	20:float,21:float,22:float,23:float}, #13, 11, 10 have bad data in them. Especially 11. It is full of garbage
	nrows=318852,
	header=0,
	parse_dates = [4],
	names=["id","Lng","Lat","Cid","tradeTime","DOM","followers",
	"totalPrice","price","square","livingRoom","drawingRoom","kitchen",
	"bathRoom","buildingType","constructionTime","renovationCondition",
	"buildingStructure","ladderRatio","elevator","fiveYearsProperty",
	"subway","district","communityAverage"])

stations = pd.read_json('data/stations.json')


housing_Beijing_with_time = raw_housing_Beijing.fillna(axis=0,method='backfill') # 12, "ConstructionTime" has one null entry!

housing_Beijing = housing_Beijing_with_time.drop(["tradeTime"], axis = 1) 

from sklearn import preprocessing

x = housing_Beijing.values
min_max_scaler = preprocessing.MinMaxScaler()
housing_Beijing_scaled = min_max_scaler.fit_transform(x)
housing_Beijing_normalized = pd.DataFrame(housing_Beijing_scaled)
# print(housing_Beijing_normalized)

print("Program data_collection ended!")

