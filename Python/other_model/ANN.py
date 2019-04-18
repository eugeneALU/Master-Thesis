import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

# read in the data
# data = pd.read_excel('RESULT.xlsx')
data = pd.read_csv('RFI_TOTAL_train.csv')
y = data['STAGE']
x = data.drop(['PID', 'STAGE', 'SLICE', 'AREA', 'NLE', 'RFI_AVG'], axis=1)
# x = data[['GLRLM_LRLGLE','GLRLM_SRLGLE','GLRLM_LGRE','GLCM_IMC1']]

data_test = pd.read_csv('RFI_TOTAL_test.csv')
y_test = data_test['STAGE']
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
Num0 = sum((Result == 0).astype(int))
Num1 = sum((Result == 1).astype(int))
# get less amount label data
data1 = x_train[Result == 1]
label1 = Result[Result == 1]
# multiply less amount label to have same amount of the label which have more ammount
Times = int(Num0/Num1) - 1
if Times > 0:
	for i in range(Times):
		x_train = np.concatenate([x_train,data1])
		Result = np.concatenate([Result,label1])  

MLP = MLPClassifier(solver='adam', alpha=1e-4, hidden_layer_sizes=(25,6), activation='relu',early_stopping=True,
		max_iter=10000, momentum=0.1,beta_1=0.55,beta_2=0.55,learning_rate='constant',power_t=0.55,tol=1e-6,random_state=1)
MLP.fit(x_train, Result)                         

# predict
y_pred = MLP.predict(x_test)
y_prob = MLP.predict_proba(x_test)  # entry 0,1 as probability for 0,1 respectively

# ROC, AUC
fpr, tpr, thresholds = roc_curve(Result_test, y_prob[:,1])
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
TP = (y_prob[Result_test==1,1]) >= thresholds[index] 
TN = (y_prob[Result_test==0,1]) < thresholds[index] 
Accuracy = (sum(TP.astype(int)) + sum(TN.astype(int))) / y_test.shape[0]
print("Threshold:{:8.5f}".format(thresholds[index]))

	# ACCU = ACCU + Accuracy
	# TPR = TPR + tpr[index]
	# SPE = SPE + specificity[index]
	# AUC = AUC + roc_auc
	# print("AUC: {:8.2f}".format(AUC/5))
	# print("Sensitivity: {:8.2f}%".format(TPR/5))
	# print("Specificity: {:8.2f}%".format(SPE/5))
	# print("Accuracy:  {:8.2f}%".format(ACCU/5))

print("AUC: {:8.2f}".format(roc_auc))
print("Sensitivity: {:8.2f}%".format(tpr[index]*100))
print("Specificity: {:8.2f}%".format(specificity[index]*100))
print("Accuracy:  {:8.2f}%".format(Accuracy*100))

##################################################
####        Grid search for best parameters
##################################################
# param_search = {
#     'beta_1' : [x*0.1 for x in np.linspace(1,10,4,endpoint=False)],
#     'beta_2' : [x*0.1 for x in np.linspace(1,10,4,endpoint=False)],
#     'activation' : ['relu'],
#     'momentum' : [x*0.1 for x in np.linspace(1,10,4,endpoint=False)],
#     'learning_rate' : ['constant', 'invscaling', 'adaptive'],
#     'power_t' : [x*0.1 for x in np.linspace(1,10,4,endpoint=False)],
#     'hidden_layer_sizes' :[(8,4), (20,5), (10,5), (20,10)],
#     'solver' : ['adam']
# }

# gsearch = GridSearchCV(MLP , param_grid = param_search, scoring='roc_auc', cv=5)
# gsearch.fit(x_train,y_train)
# print(gsearch.best_params_)
# print(gsearch.best_score_)
# print("Finish")