import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# read in Train data
data = pd.read_csv('RFI_TOTAL_train.csv')
y = data['STAGE']
x = data.drop(['PID', 'STAGE', 'SLICE', 'AREA', 'NLE', 'RFI_AVG'], axis=1)

# read in HIFI data
data = pd.read_csv('HIFI.csv')
PID = data[['PID','STAGE','AREA']]
y_test = data['STAGE']
x_test = data.drop(['PID', 'STAGE', 'SLICE', 'AREA'], axis=1)

R34 = (y > 2).astype(int)
R34_test = (y_test > 2).astype(int)

Result = np.array(R34)
Result_test = np.array(R34_test)

# Standardize the feartures
scaler = StandardScaler()
x_train = scaler.fit_transform(x)
x_test = scaler.transform(x_test)


clf= AdaBoostClassifier(n_estimators=55, learning_rate=0.005, algorithm='SAMME.R') 
clf.fit(x_train, Result)

# clf = SVC(C=5, kernel='linear', tol=0.0001, probability=True)
# clf.fit(x_train,Result)

# clf = MLPClassifier(solver='adam', alpha=1e-4, hidden_layer_sizes=(25,6), activation='relu',
# 		early_stopping=True, max_iter=10000,beta_1=0.55,beta_2=0.55,tol=1e-6) 
# clf.fit(x_train, Result)

y_pred = clf.predict(x_test)
y_prob = clf.predict_proba(x_test)

fpr, tpr, thresholds = roc_curve(Result_test, y_prob[:,1])
roc_auc = auc(fpr, tpr)
plt.figure(3)
plt.plot(fpr, tpr, lw=1, label='ROC (area = {:0.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], '--', color='r', label='Reference')
plt.legend(loc='upper left')
plt.title('ROC of stage 0-2 vs 3-4')
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
print("AUC: {:8.2f}".format(roc_auc))
print("Sensitivity: {:8.5f}%".format(sum(TP.astype(int))/(y_prob[Result_test==1,1].shape[0])*100))
print("Specificity: {:8.5f}%".format(sum(TN.astype(int))/(y_prob[Result_test==0,1].shape[0])*100))
print("Accuracy:  {:8.2f}%".format(Accuracy*100))


adjusty = y_prob[:,1] >= thresholds[index]
adjusty = adjusty.astype(int)
# print(adjusty)
print('Stage 0~2:', adjusty[Result_test<1].shape)
print('Stage 3~4:', adjusty[Result_test>0].shape)

positive = adjusty[Result_test>0]
negative = adjusty[Result_test<1]
PIDp = PID[Result_test>0]
PIDn = PID[Result_test<1]
# PIDp[positive<1].to_csv('HIFI ErrorList_p.csv', index=False)
# PIDn[negative>0].to_csv('HIFI ErrorList_n.csv', index=False)

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

