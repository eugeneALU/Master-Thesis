import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

labels = ['stage0','stage1','stage2','stage3','stage4']
data = pd.read_excel('RFIavg_oneslice.xlsx')

label = data['STAGE']
R34 = (label > 2).astype(int) # 21 samples

a = -52.399-2.055*data['NLE']+(-11.372)*data['GLRLM_LRLGLE']+52.933*data['GLCM_MAXP']+13.463*data['GLCM_SUME']
expa = np.exp(a)
RFI = expa/(1+expa)

# box: RFI & stage
y0 = RFI.loc[label==0]
y1 = RFI.loc[label==1]
y2 = RFI.loc[label==2]
y3 = RFI.loc[label==3]
y4 = RFI.loc[label==4]
y_box = [y0,y1,y2,y3,y4]

plt.figure(1)
plt.scatter(label,RFI)
plt.boxplot(y_box, labels=labels, positions=range(0,5))
plt.grid(axis='y')
plt.xlabel('Stage')
plt.ylabel('RFI')
plt.show()

# AUC RFI & STAGE
fpr, tpr, thresholds = roc_curve(R34, RFI)
specificity = 1 - fpr
roc_auc = auc(fpr, tpr)
plt.figure(2)
plt.plot(fpr, tpr, lw=1, label='ROC (area = {:0.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], '--', color='r', label='Reference')
plt.legend(loc='upper left')
plt.title('ROC of RFI(stage 0-2 vs 3-4)')
plt.show()

# find best threshold
highest = 0
index = 0
for i in range(fpr.shape[0]):
    ADD = specificity[i] + tpr[i]
    if (ADD > highest):
        highest = ADD
        index = i
TP = (RFI.loc[R34==1]) >= thresholds[index] 
TN = (RFI.loc[R34==0]) < thresholds[index] 
Accuracy = (sum(TP.astype(int)) + sum(TN.astype(int))) / RFI.shape[0]
print("Threshold:{:8.2f}".format(thresholds[index]))
print("Sensitivity: {:8.2f}%".format(tpr[index]*100))
print("Specificity: {:8.2f}%".format(specificity[index]*100))
print("Accuracy:  {:8.2f}%".format(Accuracy*100))

