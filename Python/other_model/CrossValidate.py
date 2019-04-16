import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model

# read in the data
Data = pd.read_csv('RFI_TOTAL_train.csv')
PID = Data['PID'].unique()

# cross validation
kf = KFold(n_splits=5,shuffle=True)
# Standardize the feartures
scaler = StandardScaler()

TPR = 0
SPE = 0
ACCU = 0
AUC = 0
for train_index, valid_index in kf.split(PID):          #split with PID
    Valid = pd.DataFrame()
    Train = Data
    for index in valid_index:
        Valid = Valid.append(Data[Data['PID'] == PID[index]])
        Train = Train[Train['PID'] != PID[index]]

    TrainY = Train['STAGE']
    TrainX = Train.drop(['PID', 'STAGE', 'SLICE', 'AREA', 'NLE'], axis=1)
    ValidY = Valid['STAGE']
    ValidX = Valid.drop(['PID', 'STAGE', 'SLICE', 'AREA', 'NLE'], axis=1)

    #####################################
    #######      change label
    #####################################
    R24 = (TrainY > 1).astype(int)
    R34 = (TrainY > 2).astype(int)
    R4 =  (TrainY > 3).astype(int)

    ValidR24 = (ValidY > 1).astype(int)
    ValidR34 = (ValidY > 2).astype(int)
    ValidR4 =  (ValidY > 3).astype(int)

    TrainX = np.array(TrainX)
    ValidX = np.array(ValidX)
    Result = np.array(R34)
    Result_valid = np.array(ValidR34)

    x_train = scaler.fit_transform(TrainX)
    x_valid= scaler.transform(ValidX)

    # augment training data    
    Num0 = sum((Result == 0).astype(int))
    Num1 = sum((Result == 1).astype(int))
    Weight0 = Num1/(Num0+Num1)
    Weight1 = Num0/(Num0+Num1)
    # get less amount label data
    data1 = x_train[Result == 1]
    label1 = Result[Result == 1]
    # multiply less amount label to have same amount of the label which have more ammount
    # Times = int(Num0/Num1) - 1
    # if Times > 0:
    # 	for i in range(Times):
    # 		x_train = np.concatenate((x_train,data1),axis=0)
    # 		Result = np.concatenate((Result,label1),axis=0)

    # Classifier
    # clf = AdaBoostClassifier(n_estimators=55, learning_rate=0.005, algorithm='SAMME.R') 
    # clf.fit(x_train, Result)

    # clf = MLPClassifier(solver='adam', alpha=1e-4, hidden_layer_sizes=(25,6), activation='relu',early_stopping=True,
	# 	max_iter=10000, momentum=0.1,beta_1=0.55,beta_2=0.55,learning_rate='constant',power_t=0.55,tol=1e-6,random_state=1)
    # clf.fit(x_train, Result)   

    # clf = SVC(C=5, kernel='linear',degree=4, gamma=0.001, tol=0.0001, class_weight={0:Weight0, 1:Weight1}, probability=True)
    # clf.fit(x_train,Result)

    # clf = SVC(kernel='rbf', class_weight={0:Weight0, 1:Weight1}, probability=True)
    # clf.fit(x_train,Result)

    # clf = RandomForestClassifier(n_estimators=25, min_samples_split=2, max_features='auto',
    #     min_weight_fraction_leaf=0, criterion='gini', class_weight={0:Weight0, 1:Weight1})
    # clf.fit(x_train,Result)

    # clf = linear_model.SGDClassifier(loss='log', penalty='elasticnet', alpha=0.0001, l1_ratio=0.1,
    #     tol=0.001,max_iter=10000)
    # clf.fit(x_train, Result)

    # predict
    y_pred = clf.predict(x_valid)
    y_prob = clf.predict_proba(x_valid)  # entry 0,1 as probability for 0,1 respectively

    # ROC, AUC
    fpr, tpr, thresholds = roc_curve(Result_valid, y_prob[:,1])
    roc_auc = auc(fpr, tpr)

    # find best threshold
    specificity = 1 - fpr
    highest = 0
    index = 0
    for i in range(fpr.shape[0]):
        ADD = specificity[i] + tpr[i]
        if (ADD > highest):
            highest = ADD
            index = i
    TP = (y_prob[Result_valid==1,1]) >= thresholds[index] 
    TN = (y_prob[Result_valid==0,1]) < thresholds[index] 
    Accuracy = (sum(TP.astype(int)) + sum(TN.astype(int))) / ValidY.shape[0]
    print("Threshold:{:8.5f}".format(thresholds[index]))

    ACCU = ACCU + Accuracy
    TPR = TPR + tpr[index]
    SPE = SPE + specificity[index]
    AUC = AUC + roc_auc

print("AUC: {:8.2f}".format(AUC/5))
print("Sensitivity: {:8.2f}%".format(TPR/5*100))
print("Specificity: {:8.2f}%".format(SPE/5*100))
print("Accuracy:  {:8.2f}%".format(ACCU/5*100))
