import numpy as np
import pandas as pd
import os
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

# read in the data
# data = pd.read_excel('RFIavg_oneslice_train.xlsx')
data = pd.read_csv('RFI_TOTAL_train.csv')
y = data['STAGE']
# x = data.drop(['PID', 'STAGE', 'SLICE', 'RFI_AVG', 'AREA', 'NLE'], axis=1)
x = data[['GLRLM_LRLGLE','GLRLM_SRLGLE','GLRLM_LGRE','GLCM_IMC1','GLRLM_SRHGLE','GLCM_IMC2','GLCM_DIFE']]

# data_test = pd.read_excel('RFIavg_oneslice_test.xlsx')
data_test = pd.read_csv('RFI_TOTAL_test.csv')
y_test = data_test['STAGE']
# x_test = data_test.drop(['PID', 'STAGE', 'SLICE', 'RFI_AVG', 'AREA', 'NLE'], axis=1)
x_test = data_test[['GLRLM_LRLGLE','GLRLM_SRLGLE','GLRLM_LGRE','GLCM_IMC1','GLRLM_SRHGLE','GLCM_IMC2','GLCM_DIFE']]


#####################################
#######      change label
#####################################
R24 = (y > 1).astype(int) # 46 samples
R34 = (y > 2).astype(int) # 21 samples
R4 =  (y > 3).astype(int) # 7 samples

R24_test = (y_test > 1).astype(int)
R34_test = (y_test > 2).astype(int)
R4_test =  (y_test > 3).astype(int)

x = np.array(x)
Result = np.array(R34)
Result_test = np.array(R34_test)

# Standardize the feartures
scaler = StandardScaler()

# Standardize the feartures
x_train = scaler.fit_transform(x)
x_test= scaler.transform(x_test)

# augment training data    
Num0 = sum((Result == 0).astype(int))
Num1 = sum((Result == 1).astype(int))
# Num3 = sum((Result == 3).astype(int))
# Num4 = sum((Result == 4).astype(int))
# get less amount label data
data1 = x_train[Result == 1]
label1 = Result[Result == 1]
# multiply less amount label to have same amount of the label which have more ammount
Times = int(Num0/Num1) - 1
if Times > 0:
    for i in range(Times):
        x_train = np.concatenate((x_train,data1),axis=0)
        Result = np.concatenate((Result,label1),axis=0)

# model -- logistic regression with elasticnet
regr = linear_model.SGDClassifier(loss='log', penalty='elasticnet', alpha=0.0001, l1_ratio=0.1,
        max_iter=10000, tol=0.001)

# train 
regr.fit(x_train, Result)

# predict
y_pred = regr.predict(x_test)
y_prob = regr.predict_proba(x_test)  # entry 0,1 as probability for 0,1 respectively

# ROC, AUC
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
Accuracy = (sum(TP.astype(int)) + sum(TN.astype(int))) / Result_test.shape[0]
print("Threshold:{:8.5f}".format(thresholds[index]))
print("Sensitivity: {:8.2f}%".format(tpr[index]*100))
print("Specificity: {:8.2f}%".format(specificity[index]*100))
# Accuracy 
# print("Origin Accuracy(TH = 0.5): %f" % accuracy_score(Result_test, y_pred))
print("Accuracy:  {:8.2f}%".format(Accuracy*100))
print("Number of Positive sample: ", sum(Result_test))

# prob
# for i in range(y_prob.shape[0]):
#         print("Test{:2d} Probability to be 1: {:.2f}".format(i,y_prob[i,1]))

# regressor coefficient 
####################################
#### a = b0 + b1 * f1 + ....
#### P(y=1| F) = exp(a)/[1+exp(a)]
####################################
print('Coefficients(bias aka constant term): \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)
# >>> Coefficients: [ 2.]
