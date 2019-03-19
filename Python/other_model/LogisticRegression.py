import numpy as np
import pandas as pd
import os
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

# run through all the directory (include sub directory)
# for root, dirs, files in os.walk("."):
#     for filename in files:
#         a = 2

# read in the data
data = pd.read_csv('RFI_TOTAL_train.csv')
y = data['STAGE']
x = data.drop(['PID', 'STAGE', 'SliceNum', 'AREA'], axis=1)
x_select = data[['GLRLM_LRLGLE','GLRLM_SRLGLE','GLRLM_LGRE','GLCM_IMC1']]
#x = data.drop(['PID', 'STAGE', 'SliceNum', 'AREA', 'NLE'], axis=1)

#####################################
#######      change label
#####################################
R24 = (y > 1).astype(int) # 46 samples
R34 = (y > 2).astype(int) # 21 samples
R4 =  (y > 3).astype(int) # 7 samples

# Standardize the feartures
std_x = preprocessing.scale(x_select, axis= 1)
std_x = pd.DataFrame(std_x)

# split data
x_train, x_test, y_train, y_test = train_test_split(std_x, R34, test_size=0.25, random_state=1)

# augment training data    
Num0 = sum((y_train == 0).astype(int))
Num1 = sum((y_train == 1).astype(int))
# Num3 = sum((y_train == 3).astype(int))
# Num4 = sum((y_train == 4).astype(int))
# get less amount label data
data1 = x_train.loc[y_train == 1]
label1 = y_train.loc[y_train == 1]
# multiply less amount label to have same amount of the label which have more ammount
Times = int(Num0/Num1) - 1
if Times > 0:
    for i in range(Times):
        x_train = pd.concat([x_train,data1])
        y_train = pd.concat([y_train,label1])   

# model // logistic regression with elasticnet
regr = linear_model.SGDClassifier(loss='log', penalty='elasticnet', alpha=0.0001, l1_ratio=0.1,
        max_iter=10000, tol=0.001)

# train 
regr.fit(x_train, y_train)

# predict
y_pred = regr.predict(x_test)
y_prob = regr.predict_proba(x_test)  # entry 0,1 as probability for 0,1 respectively

# ROC, AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob[:,1])
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
TP = (y_prob[y_test==1,1]) >= thresholds[index] 
TN = (y_prob[y_test==0,1]) < thresholds[index] 
Accuracy = (sum(TP.astype(int)) + sum(TN.astype(int))) / y_test.shape[0]
print("Threshold:{:8.5f}".format(thresholds[index]))
print("Sensitivity: {:8.2f}%".format(tpr[index]*100))
print("Specificity: {:8.2f}%".format(specificity[index]*100))
# Accuracy 
print("Origin Accuracy(TH = 0.5): %f" % accuracy_score(y_test, y_pred))
print("Accuracy:  {:8.2f}%".format(Accuracy*100))

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
