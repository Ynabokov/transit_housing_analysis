import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import os
import datetime
import seaborn as sns

from data_collection import housing_Beijing
from get_stations import get_stations
from data_collection import stations
from data_preprocessing import stations_locations
from data_preprocessing import housing_Beijing

print(housing_Beijing.info())
print("----------------------------------------")
print(housing_Beijing.describe()) #Have a lot of categorical data
print("----------------------------------------")
housing_Beijing.hist(bins=50,figsize=(20,15))
plt.show()


plt.figure(figsize=(8,6))
sns.set_context('paper', font_scale=1.0)

data_corr = housing_Beijing.corr()
sns.heatmap(data_corr, annot=True, cmap="Spectral")
plt.show()


important_data = housing_Beijing.drop(["DOM",
	"livingRoom","drawingRoom","kitchen",
	"bathRoom","buildingType","constructionTime","renovationCondition",
	"buildingStructure","ladderRatio","elevator",
	"subway","district","communityAverage"],axis = 1)

#visualize geo-data
plt.scatter(stations_locations.iloc[:,1],stations_locations.iloc[:,2])
plt.show()

#visualize geo-data
stn = get_stations()
ax = important_data.plot(kind="scatter", x="Lng", y="Lat", alpha=0.4, #Idea taken from the "Hands-On Machine Learning with Scikit-Learn" book
	s=important_data["totalPrice"]/100, label="total price", figsize=(10,7),
	c="price", cmap=plt.get_cmap("jet"))
plt.plot(stn['Lng'], stn['Lat'], 'r.' , markersize=7,mew=7, label="Stations")
plt.legend()
ax.tick_params(labelbottom=True)
ax.set_xlabel("Longitude", fontsize = 15)
ax.set_ylabel("Latitude", fontsize = 15)
plt.savefig('map-part.png', bbox_inches = "tight")
plt.savefig(fname="transparent_geo_data",transparent=True)




""" #DO NOT RUN THE CODE BELLOW, UNLESS YOU HAVE AN HOUR TO SPARE
g = sns.PairGrid(important_data) #ORIGINAL DATA SET IS TOO BIG FOR A FULL DIPLAY (or takes too much time)
g.map_upper(sns.scatterplot)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot)
plt.show()
"""
