import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# read in the data
data = pd.read_excel('RFIavg_oneslice.xlsx')
y = data['STAGE']
NLE = data['NLE']

#####################################
#######      change label
#####################################
# change low stage to Positive Samples (different from other code)
R14 = (y <= 0).astype(int) # 46 samples
R24 = (y <= 1).astype(int) # 46 samples
R34 = (y <= 2).astype(int) # 21 samples
R4 =  (y <= 3).astype(int) # 7 samples

STAGE1 = '0123'
STAGE2 = '4'
test = R4

# scatter: NLE & STAGE
# box: NLE & stage
# label = ['stage0','stage1','stage2','stage3','stage4']
# y0 = np.array(NLE.loc[y==0])
# y1 = np.array(NLE.loc[y==1])
# y2 = np.array(NLE.loc[y==2])
# y3 = np.array(NLE.loc[y==3])
# y4 = np.array(NLE.loc[y==4])
# y_box = np.array([y0,y1,y2,y3,y4])

# plt.figure(1)
# plt.scatter(y,NLE)
# plt.boxplot(y_box, positions=range(0,5),labels=label)
# plt.xlabel('Stage')
# plt.ylabel('NLE')
# plt.show()

# ROC, AUC
fpr, tpr, thresholds = roc_curve(test, NLE)
roc_auc = auc(fpr, tpr)
# plt.figure(3)
# plt.plot(fpr, tpr, lw=1, label='ROC (area = {:0.2f})'.format(roc_auc))
# plt.plot([0, 1], [0, 1], '--', color='r', label='Reference')
# plt.legend(loc='upper left')
# plt.title('ROC of RFI(stage F{:s} vs F{:s})'.format(STAGE1,STAGE2))
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
TP = (NLE[test==1]) >= thresholds[index] 
TN = (NLE[test==0]) < thresholds[index] 
Accuracy = (sum(TP.astype(int)) + sum(TN.astype(int))) / test.shape[0]
print("Threshold:{:8.5f}".format(thresholds[index]))
print("Sensitivity: {:8.2f}%({:d}/{:d})".format(tpr[index]*100, sum(TP.astype(int)), (NLE[test==1]).shape[0]))
print("Specificity: {:8.2f}%({:d}/{:d})".format(specificity[index]*100, sum(TN.astype(int)), (NLE[test==0]).shape[0]))
# Accuracy 
print("Accuracy:  {:8.2f}%".format(Accuracy*100))