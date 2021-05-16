# CMPT353 Transit Housing Project

This is a CMPT 353 (Fall 2020) final project.

### Data Sources *(saved in data directory)*

- Housing price data: https://www.kaggle.com/ruiqurm/lianjia
- Stations data: https://www.wikidata.org

### Required Libraries

- os
- pandas
- numpy
- sys
- qwikidata
- json
- matplotlib
- seaborn
- haversine
- tqdm
- scipy
- sklearn

###Libraies for Machine Learning
-sklearn.cluster.KMeans
-sklearn.cluster.SpectralClustering
-sklearn.cluster.DBSCAN
-sklearn.preprocessing.FunctionTransformer
-sklearn.naive_bayes.GaussianNB
-sklearn.neighbors.KNeighborsClassifier
-sklearn.ensemble.RandomForestClassifier
-klearn.naive_bayes.MultinomialNB
-sklearn.tree.DecisionTreeClassifier
-sklearn.svm.SVC, SVR, NuSVR, NuSVC
-sklearn.linear_model.SGDClassifier
-sklearn.cluster.MiniBatchKMeans


### Code Structure

    .
    ├── data_illustration.py  
    ├── data                    # Raw and preprocessed data, plots
    │   ├── stat                # Data preprocessed for stat tests 
    ├── calculate_statistics.py
    ├── cluster_models.py
    ├── data_collection.py
    ├── get_stations.py
    ├── data_illustration.py
    └── README.md


### To actually run the project use following commands:


1. To run statistical tests use
> python3 calculate_statistics.py
2. To get Machine Learning results  use
> python3 cluster_models.py

***Note***
Files will be created **only** if they are missing in the data directory (or the whole directory is missing). It is not expected though that files are regenerated because it may take some decent time. This did not apply to plots - they are generated pretty quickly.
 
 All the datafiles should be in "data" repository
