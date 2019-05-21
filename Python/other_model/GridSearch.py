import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model

# read in the data
Data = pd.read_csv('RFI_TOTAL_train.csv')
PID = Data['PID'].unique()
# Standardize the feartures
scaler = StandardScaler()

TrainY = Train['STAGE']
TrainX = Train.drop(['PID', 'STAGE', 'SLICE', 'AREA', 'NLE', 'RFI_AVG'], axis=1)
#####################################
#######      change label
#####################################
R24 = (TrainY > 1).astype(int)
R34 = (TrainY > 2).astype(int)
R4 =  (TrainY > 3).astype(int)


##################################################
####        Grid search for best parameters
##################################################
param_search = {
    'n_estimators':[40,45,50,55,60], 
    'learning_rate':[x*0.1 for x in np.linspace(1,10,9,endpoint=False)],
    'algorithm':['SAMME.R','SAMME']
}

gsearch = GridSearchCV(CLF, param_grid = param_search, scoring='roc_auc', cv=5)
gsearch.fit(x_train,y_train)
print(gsearch.best_params_)
print(gsearch.best_score_)
print("Finish")