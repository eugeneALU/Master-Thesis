import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

label = ['stage0','stage1','stage2','stage3','stage4']
# read in the data
data = pd.read_csv('RFI_TOTAL_train.csv')
y = data['STAGE']
# x = data[['GLRLM_LRLGLE','GLRLM_SRLGLE','GLRLM_LGRE','GLCM_IMC1']]
x = data.drop(['PID', 'STAGE', 'SLICE', 'AREA', 'NLE', 'RFI_AVG'], axis=1)

data_test = pd.read_csv('RFI_TOTAL_test.csv')
y_test = data_test['STAGE']
# x_test = data_test[['GLRLM_LRLGLE','GLRLM_SRLGLE','GLRLM_LGRE','GLCM_IMC1']]
x_test = data_test.drop(['PID', 'STAGE', 'SLICE', 'AREA', 'NLE', 'RFI_AVG'], axis=1)

#####################################
#######      change label
#####################################
R24 = (y > 1).astype(int)
R34 = (y > 2).astype(int)
R4 =  (y > 3).astype(int)

R24_test = (y_test > 1).astype(int)
R34_test = (y_test > 2).astype(int)
R4_test =  (y_test > 3).astype(int) 

x = np.array(x)
Result = np.array(R34)
Result_test = np.array(R34_test)

# cross validation
kf = KFold(n_splits=5,shuffle=True)

# Standardize the feartures
scaler = StandardScaler()

# TPR = 0
# SPE = 0
# ACCU = 0
# AUC = 0
# for train_index, test_index in kf.split(x, Result):
	## split data
	# x_train_pre, x_test_pre = x[train_index], x[test_index]
	# y_train, y_test = Result[train_index], Result[test_index]
	## Standardize the feartures
	# x_train = scaler.fit_transform(x_train_pre)
	# x_test= scaler.transform(x_test_pre)

x_train = scaler.fit_transform(x)
x_test= scaler.transform(x_test)

# augment training data    
Num0 = sum((Result == 0).astype(int))
Num1 = sum((Result == 1).astype(int))
Weight0 = Num1/(Num0+Num1)
Weight1 = Num0/(Num0+Num1)
# get less amount label data
# data1 = x_train[y_train == 1]
# label1 = y_train[y_train == 1]
# multiply less amount label to have same amount of the label which have more ammount
# Times = int(Num0/Num1) - 1

print("Training Start")
#####################################
#######      Classifier
#####################################
clf = SVC(C=5, kernel='linear',degree=4, gamma=0.001, tol=0.0001, class_weight={0:Weight0, 1:Weight1}, probability=True)
clf.fit(x_train,Result)

y_pred = clf.predict(x_test)
y_prob = clf.predict_proba(x_test) #only with probability=True
y_dist = clf.decision_function(x_test)

# ROC, AUC
fpr, tpr, thresholds = roc_curve(Result_test, y_prob[:,1], pos_label=1)
roc_auc = auc(fpr, tpr)
# plt.figure(3)
# plt.plot(fpr, tpr, lw=1, label='ROC (area = {:0.2f})'.format(roc_auc))
# plt.plot([0, 1], [0, 1], '--', color='r', label='Reference')
# plt.legend(loc='upper left')
# plt.title('ROC of RFI(stage 0-2 vs 3-4)')
# plt.show()

# find best threshold
specificity = 1 - fpr
highest = 0
index = 0
for i in range(fpr.shape[0]):
	ADD = specificity[i] + tpr[i]
	if (ADD > highest):
			highest = ADD
			index = i
TP = (y_dist[Result_test==1]) >= thresholds[index] 
TN = (y_dist[Result_test==0]) < thresholds[index] 
Accuracy = (sum(TP.astype(int)) + sum(TN.astype(int))) / y_test.shape[0]
print("Threshold:{:8.5f}".format(thresholds[index]))

# ACCU = ACCU + Accuracy
# TPR = TPR + tpr[index]
# SPE = SPE + specificity[index]
# AUC = AUC + roc_auc

print("AUC: {:8.2f}".format(roc_auc))
print("Sensitivity: {:8.2f}%".format(tpr[index]*100))
print("Specificity: {:8.2f}%".format(specificity[index]*100))
print("Accuracy:  {:8.2f}%".format(Accuracy*100))


##################################################
####        Grid search for best parameters
##################################################
# param_search = {
#     'C':[x for x in np.linspace(1,11,10,endpoint=False)],
#     'kernel':['linear', 'rbf', 'poly'],
#     'gamma':[1,0.1,0.01,0.001,0.0001],
#     'degree':[2,3,4,5]
# }

# gsearch = GridSearchCV(clf , param_grid = param_search, scoring='roc_auc', cv=5, refit=True)
# gsearch.fit(x_train,y_train)
# print(gsearch.best_params_)
# print(gsearch.best_score_)
# print(gsearch.best_estimator_)
# print("Finish")

##################################################
#### dual_coef_ == alpha in SVM dual problem
##################################################
# weight(= coef_[i]) = sum(dual_coef_[i] * feature[i] * y) for each support vector
# EX: classifier 0 vs 1 : coef_ only decided by the weight between 0, 1 and support vector of class 0,1
# D = clf.dual_coef_
# for i in range(clf.support_.shape[0]):
#     print('{:d}'.format(y_train.iloc[clf.support_[i]]))

# C = np.zeros(shape=(1,4))
# for i in range(clf.support_.shape[0]):
#     if y_train.iloc[clf.support_[i]] == 0:
#         C = C + D[0,i] * clf.support_vectors_[i] 
#     elif y_train.iloc[clf.support_[i]] == 1:
#         C = C + D[0,i] * clf.support_vectors_[i]
##  C will = clf.coef_[0] (aka clasifier 0 vs 1)