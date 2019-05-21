import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

# read in the data
data = pd.read_csv('RFI_TOTAL_train.csv')
y = data['STAGE']
x = data.drop(['PID', 'STAGE', 'SLICE', 'AREA', 'NLE', 'RFI_AVG'], axis=1)
# x = data[['GLRLM_LRLGLE','GLRLM_SRLGLE','GLRLM_LGRE','GLCM_IMC1']]

data_test = pd.read_csv('RFI_TOTAL_test.csv')
y_test = data_test['STAGE']
AREA = data_test['AREA']
PID = data_test[['PID','STAGE','AREA']]
x_test = data_test.drop(['PID', 'STAGE', 'SLICE', 'AREA', 'NLE', 'RFI_AVG'], axis=1)
# x_test = data_test[['GLRLM_LRLGLE','GLRLM_SRLGLE','GLRLM_LGRE','GLCM_IMC1']]


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
# kf = KFold(n_splits=5,shuffle=True)

# Standardize the feartures
scaler = StandardScaler()

# TPR = 0
# SPE = 0
# ACCU = 0
# AUC = 0
# for train_index, test_index in kf.split(x, Result):
# 	# split data
# 	x_train_pre, x_test_pre = x[train_index], x[test_index]
# 	y_train, y_test = Result[train_index], Result[test_index]

# 	# Standardize the feartures
# 	x_train = scaler.fit_transform(x_train_pre)
# 	x_test= scaler.transform(x_test_pre)

x_train = scaler.fit_transform(x)
x_test= scaler.transform(x_test)

# augment training data    
# Num0 = sum((Result == 0).astype(int))
# Num1 = sum((Result == 1).astype(int))
# get less amount label data
# data1 = x_train[Result == 1]
# label1 = Result[Result == 1]
# multiply less amount label to have same amount of the label which have more ammount
# Times = int(Num0/Num1) - 1
# if Times > 0:
# 	for i in range(Times):
# 		x_train = np.concatenate((x_train,data1),axis=0)
# 		Result = np.concatenate((Result,label1),axis=0)

ADA= AdaBoostClassifier(n_estimators=55, learning_rate=0.005, algorithm='SAMME.R') 
ADA.fit(x_train, Result)

# predict
y_pred = ADA.predict(x_test)
y_prob = ADA.predict_proba(x_test)  # entry 0,1 as probability for 0,1 respectively
# y_pred = ADA.predict(x_train)
# y_prob = ADA.predict_proba(x_train) 

# Area filter
# y_prob = y_prob[AREA>10000]
# Result_test = Result_test[AREA>10000]

# ROC, AUC
# fpr, tpr, thresholds = roc_curve(Result, y_prob[:,1])
fpr, tpr, thresholds = roc_curve(Result_test, y_prob[:,1])
roc_auc = auc(fpr, tpr)
plt.figure(3)
plt.plot(fpr, tpr, lw=1, label='ROC (area = {:0.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], '--', color='r', label='Reference')
plt.legend(loc='upper left')
plt.title('ROC of RFI(stage 0-2 vs 3-4)')
plt.show()

# find best threshold
specificity = 1 - fpr
highest = 0
index = 0
for i in range(fpr.shape[0]):
	ADD = specificity[i] + tpr[i]
	if (ADD > highest):
		highest = ADD
		index = i
TP = (y_prob[Result_test==1,1]) >= thresholds[index]
TN = (y_prob[Result_test==0,1]) < thresholds[index]
Accuracy = (sum(TP.astype(int)) + sum(TN.astype(int))) / y_test.shape[0]
print("Threshold:{:8.5f}".format(thresholds[index]))

# ACCU = ACCU + Accuracy
# TPR = TPR + tpr[index]
# SPE = SPE + specificity[index]
# AUC = AUC + roc_auc

print("AUC: {:8.2f}".format(roc_auc))
print("Sensitivity: {:8.5f}%".format(sum(TP.astype(int))/(y_prob[Result_test==1,1].shape[0])))
print("Specificity: {:8.5f}%".format(sum(TN.astype(int))/(y_prob[Result_test==0,1].shape[0])))
print("Accuracy:  {:8.2f}%".format(Accuracy*100))

##################################################
####        Grid search for best parameters
##################################################
# param_search = {
#     'n_estimators':[40,45,50,55,60], 
#     'learning_rate':[x*0.1 for x in np.linspace(1,10,10,endpoint=False)],
#     'algorithm':['SAMME.R','SAMME']
# }

# gsearch = GridSearchCV(ADA , param_grid = param_search, scoring='roc_auc', cv=5)
# gsearch.fit(x_train, Result)
# print(gsearch.best_params_)
# print(gsearch.best_score_)
# print("Finish")

##################################################
####        Comfusion matrix
##################################################
y_prob_test = y_prob
# y_prob_test = ADA.predict_proba(x_test)
adjusty = y_prob_test[:,1] >= thresholds[index]
adjusty = adjusty.astype(int)
# print(adjusty)
print('Stage 0~2:', adjusty[Result_test<1].shape)
print('Stage 3~4:', adjusty[Result_test>0].shape)
# print('Stage 0~2:', y_pred[Result_test<1].shape)
# print('Stage 3~4:', y_pred[Result_test>0].shape)
positive = adjusty[Result_test>0]
negative = adjusty[Result_test<1]
PIDp = PID[Result_test>0]
PIDn = PID[Result_test<1]
PIDp[positive<1].to_csv('ErrorList_p.csv', index=False)
PIDn[negative>0].to_csv('ErrorList_n.csv', index=False)
cm = confusion_matrix(Result_test, adjusty)
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
ax.figure.colorbar(im, ax=ax)
classes = ['Stage0~2', 'Stage3~4']
ax.set(xticks=np.arange(cm.shape[1]),
	yticks=np.arange(cm.shape[0]),
	xticklabels=classes, yticklabels=classes,
	title='Confusion matrix with Stage0~2 VS Stage3~4',
	ylabel='True label',
	xlabel='Predicted label')
fmt = 'd'
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
	for j in range(cm.shape[1]):
		ax.text(j, i, format(cm[i, j], fmt),
				ha="center", va="center",
				color="white" if cm[i, j] > thresh else "black")
fig.tight_layout()
plt.show()