import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabaz_score

# read in the data
# data = pd.read_excel('RESULT.xlsx')
data = pd.read_csv('RFI_TOTAL_train.csv')
label = ['stage0','stage1','stage2','stage3','stage4']
y = data['STAGE']
# x = data[['GLRLM_LRLGLE','GLRLM_SRLGLE','GLRLM_LGRE','GLCM_IMC1']]
x = data.drop(['PID', 'STAGE', 'SLICE', 'AREA', 'NLE'], axis=1)

#####################################
#######      change label
#####################################
R24 = (y > 1).astype(int) # 46 samples
R34 = (y > 2).astype(int) # 21 samples
R4 =  (y > 3).astype(int) # 7 samples

# Standardize the feartures
std_x = preprocessing.scale(x, axis = 0)
std_x = pd.DataFrame(std_x)


# augment training data    
Num0 = sum((R34 == 0).astype(int))
Num1 = sum((R34 == 1).astype(int))
Weight0 = Num1/(Num0+Num1)
Weight1 = Num0/(Num0+Num1)
# get less amount label data
data1 = x.loc[R34 == 1]
label1 = y.loc[R34 == 1]
# multiply less amount label to have same amount of the label which have more ammount
Times = int(Num0/Num1) - 1
if Times > 0:
    for i in range(Times):
        x = pd.concat([x,data1])
        R34 = pd.concat([R34,label1])

#####################################
#######      Classifier
#####################################
n_clusters = 3
clf = KMeans(n_clusters= n_clusters, init='random', n_init=10, max_iter=1000)
clf.fit(x)

#y_pred = clf.predict(x)
y_pred = clf.labels_
J = clf.inertia_
center = clf.cluster_centers_

print("Origin Accuracy: %f" % accuracy_score(R34, y_pred))

silhouette = silhouette_score(x, y_pred)
calinski_harabaz = calinski_harabaz_score(x, y_pred)
print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette)
print("For n_clusters =", n_clusters, "The average calinski_harabaz_score is :", calinski_harabaz)

##################################################
####        Grid search for best parameters
##################################################
# param_search = {
#     'n_clusters': [2,3,4],
#     'init': ['k-means++', 'random']
# }

# gsearch = GridSearchCV(clf , param_grid = param_search, scoring='accuracy', cv=5)
# gsearch.fit(x,R34)
# print(gsearch.best_params_)
# print(gsearch.best_score_)
# print("Finish")