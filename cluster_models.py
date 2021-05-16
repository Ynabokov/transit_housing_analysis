import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from data_preprocessing import X_train, X_valid, y_train, y_valid, X
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from data_preprocessing import stations_locations
import math
from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.pipeline import make_pipeline

from sklearn.naive_bayes import BernoulliNB
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import FunctionTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, SVR, NuSVR, NuSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import PassiveAggressiveRegressor

#Standart models
model_k_means = make_pipeline(
	MinMaxScaler(),
	KMeans(n_clusters = 7, init="k-means++")
	)

model_spectral = make_pipeline(
	MinMaxScaler(),
	SpectralClustering(3, affinity='precomputed', n_init=100, assign_labels='discretize')
	)

model_dbscan = make_pipeline(
	MinMaxScaler(),
	DBSCAN(eps=0.3, min_samples=10)
	)

#Custom model - inspired by an example from the notes
def find_closes_statyion(x):
	Lng = x[0]
	Lat = x[1]
	min_dist = math.sqrt((stations_locations.iloc[0,1] - Lng)**2 +
						 (stations_locations.iloc[0,2] - Lat)**2)
	closets_station = stations_locations.iloc[0,1], stations_locations.iloc[0,2]
	for station in stations_locations.values:
		dist = math.sqrt((station[1] - Lng)**2 +
						 (station[2] - Lat)**2)
		if dist < min_dist:
			min_dist = dist
			closets_station_x = station[1]
			closets_station_y = station[2]
	return [min_dist, closets_station_x, closets_station_y]

def add_nearest_station_stations(X):
	result = list()
	for x in X:
		result.append(find_closes_statyion(x))	
	return result

model_custom = make_pipeline(
	StandardScaler(),
	FunctionTransformer(add_nearest_station_stations, validate=True),
	GaussianNB()
	)

model_k_neighbours = make_pipeline(
	MinMaxScaler(),
	KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=2)
	)

model_random_forest = make_pipeline(
	StandardScaler(),
	RandomForestClassifier(n_estimators=50, max_depth=4, min_samples_leaf=300)
	)

model_svm = make_pipeline(
	StandardScaler(),
	NuSVR()
	)

model_tree = make_pipeline(
	StandardScaler(),
	DecisionTreeClassifier(criterion='entropy', random_state=0)
	)


print("K Neighbors:")
model_k_neighbours.fit(X_train, y_train)
y_pred_neighbors = model_k_neighbours.predict(X_valid)
print(str(accuracy_score(y_valid, y_pred_neighbors)*100) + " out of 100")

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_valid = sc.transform(X_valid)

#Mini-Batch Models
model_MultinomialNB = MultinomialNB(alpha=0.01)

model_SGDClassifier = SGDClassifier(shuffle=True, loss='log')

model_MiniBatchKMeans = MiniBatchKMeans()


#Attempted to solve the memmory problem with mini-batches. Did not help
num_batches = 100
size_of_batch = 1689
prev_num = 0
iterator = size_of_batch
all_classes = np.array([0, 1])
while iterator < len(X_train):
	print("Batch #" + str(iterator/size_of_batch) + " out of 100")
	
	model_SGDClassifier.partial_fit(X_train[prev_num:iterator, :], y_train[prev_num:iterator],
		classes=np.unique(y_train))
	prev_num = iterator
	iterator += size_of_batch

print("SGDClassifier:")
y_pred_SGDClassifier = model_SGDClassifier.predict(X_valid)
print(str(accuracy_score(y_valid, y_pred_SGDClassifier)*100) + " out of 100")



#Sample of the models which were tried, but crashed, due to a memmory problem
""" 
print("MiniBatchKMeans:")
y_pred_MiniBatchKMeans = model_MiniBatchKMeans.predict(X_valid)
print(str(accuracy_score(y_valid, y_pred_MiniBatchKMeans)*100) + " out of 100")

print("Perceptron:")
y_pred_Perceptron = model_Perceptron.predict(X_valid)
print(str(accuracy_score(y_valid, y_pred_Perceptron)*100) + " out of 100")

print("MultinomialNB:")
y_pred_MultinomialNB = model_MultinomialNB.predict(X_valid)
print(str(accuracy_score(y_valid, y_pred_MultinomialNB)*100) + " out of 100")

print("Custom model:")
model_custom.fit(X_train, y_train)
y_pred_custom = model_custom.predict(X_valid)
print(str(accuracy_score(y_valid, y_pred_custom)*100) + " out of 100")

attempts = list()
for i in range(1, 20): #Have foundthis similar code in the pile of my old code from online courses I have taken last year. Original Credit: SuperDataScience on Udemy
	kmeans = KMeans(n_clusters = i, init="k-means++", random_state = 0)
	kmeans.fit(housing_Beijing_normalized.values)
	attempts.append(kmeans.inertia_) #Inertia: Sum of squared distances of samples to their closest cluster center.

plt.plot(range(1,20), attempts) #4 or 5 clysters are optimal
plt.title("Elbow estiamte")
plt.xlabel("# of clusters")
plt.ylabel("Inertia")
plt.show()
"""
#kmeans = Means(nclusters = 5, init="k-means++")