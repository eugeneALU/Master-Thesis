import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

# read in the data
data = pd.read_excel('RFIavg_oneslice.xlsx')
label = ['stage0','stage1','stage2','stage3','stage4']
y = data['STAGE']
x = data[['GLRLM_LRLGLE','GLRLM_SRLGLE','GLRLM_LGRE','GLCM_IMC1']]
# x = data.drop(['PID', 'STAGE', 'SliceNum', 'AREA', 'NLE'], axis=1)

#####################################
#######      change label
#####################################
R24 = (y > 1).astype(int) # 46 samples
R34 = (y > 2).astype(int) # 21 samples
R4 =  (y > 3).astype(int) # 7 samples

# Standardize the feartures
std_x = preprocessing.scale(x, axis= 1)
std_x = pd.DataFrame(std_x)

# split data
x_train, x_test, y_train, y_test = train_test_split(x, R34, test_size=0.25, random_state=1)

# augment training data    
Num0 = sum((y_train == 0).astype(int))
Num1 = sum((y_train == 1).astype(int))
Weight0 = Num1/(Num0+Num1)
Weight1 = Num0/(Num0+Num1)
# get less amount label data
data1 = x_train.loc[y_train == 1]
label1 = y_train.loc[y_train == 1]
# multiply less amount label to have same amount of the label which have more ammount
Times = int(Num0/Num1) - 1
# if Times > 0:
#     for i in range(Times):
#         x_train = pd.concat([x_train,data1])
#         y_train = pd.concat([y_train,label1])

#####################################
#######      Classifier
#####################################
clf = RandomForestClassifier(n_estimators=25, min_samples_split=2, max_features='auto',
        min_weight_fraction_leaf=0, criterion='gini', class_weight={0:Weight0, 1:Weight1})

clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)
y_prob = clf.predict_proba(x_test) # entry 0,1 as probability for 0,1 respectively

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

##################################################
####        Grid search for best parameters
##################################################
# param_search = {
#     'n_estimators':[5,10,15,20,25,30,35,40,45,50],
#     'min_samples_split': [2,3,4,5],
# 	'max_features':[None, 'auto', 0.5, 4],
# 	'criterion' :['gini','entropy']
# }

# gsearch = GridSearchCV(clf , param_grid = param_search, scoring='roc_auc', cv=5)
# gsearch.fit(x_train,y_train)
# print(gsearch.best_params_)
# print(gsearch.best_score_)
# print("Finish")