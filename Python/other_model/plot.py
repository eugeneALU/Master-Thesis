import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
from sklearn.metrics import roc_curve, auc

# read in the data
data = pd.read_excel('RFIavg_oneslice.xlsx')
label = ['stage0','stage1','stage2','stage3','stage4']
# stage 0 : 29
# stage 1 : 16
# stage 2 : 25
# stage 3 : 14
# stage 4 : 7
x = data['STAGE']

#####################################
#######      change label
#####################################
R24 = (x > 1).astype(int) # 46 samples
R34 = (x > 2).astype(int) # 21 samples
R4 =  (x > 3).astype(int) # 7 samples

#####################################
########        kendall
#####################################
y = data.drop(['PID', 'STAGE', 'SliceNum', 'AREA'], axis=1)
for i in range(y.shape[1]):
    feature = y.iloc[:, i]
    tau, p_value = kendalltau(feature,R4) #x, R34, R24
    print("{:<12} kendalltau is: {:20.15f}".format(feature.name,tau))

#####################################
#######      RFI
#####################################
y = data['RFI_Avg']
# scatter: RFI & STAGE
plt.figure(1)
plt.scatter(x,y)
plt.xticks([0,1,2,3,4])
plt.xlabel('Stage')
plt.ylabel('RFI')
plt.show()

# box: RFI & stage
y0 = y.loc[x==0]
y1 = y.loc[x==1]
y2 = y.loc[x==2]
y3 = y.loc[x==3]
y4 = y.loc[x==4]
y_box = [y0,y1,y2,y3,y4]

plt.figure(2)
plt.boxplot(y_box, labels=label)
plt.grid(axis='y')
plt.xlabel('Stage')
plt.ylabel('RFI')
plt.show()

# AUC RFI & STAGE
fpr, tpr, thresholds = roc_curve(R24, y)
specificity = 1 - fpr
roc_auc = auc(fpr, tpr)
plt.figure(3)
plt.plot(fpr, tpr, lw=1, label='ROC (area = {:0.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], '--', color='r', label='Reference')
plt.legend(loc='upper left')
plt.title('ROC of RFI(stage 0-1 vs 2-4)')
plt.show()

# find best threshold
highest = 0
index = 0
for i in range(fpr.shape[0]):
    ADD = specificity[i] + tpr[i]
    if (ADD > highest):
        highest = ADD
        index = i
TP = (y.loc[R24==1]) >= thresholds[index] 
TN = (y.loc[R24==0]) < thresholds[index] 
Accuracy = (sum(TP.astype(int)) + sum(TN.astype(int))) / y.shape[0]
print("Threshold:{:8.2f}".format(thresholds[index]))
print("Sensitivity: {:8.2f}%".format(tpr[index]*100))
print("Specificity: {:8.2f}%".format(specificity[index]*100))
print("Accuracy:  {:8.2f}%".format(Accuracy*100))

#####################################
#######      NLE
#####################################
y = data['NLE']
# scatter: NLE & STAGE
plt.figure(1)
plt.scatter(x,y)
plt.xticks([0,1,2,3,4])
plt.xlabel('Stage')
plt.ylabel('NLE')
plt.show()

# box: NLE & stage
y0 = y.loc[x==0]
y1 = y.loc[x==1]
y2 = y.loc[x==2]
y3 = y.loc[x==3]
y4 = y.loc[x==4]
y_box = [y0,y1,y2,y3,y4]

plt.figure(2)
plt.boxplot(y_box, labels=label)
plt.grid(axis='y')
plt.xlabel('Stage')
plt.ylabel('NLE')
plt.show()

#####################################
########       GLRLM_LRLGLE
#####################################
y = data['GLRLM_LRLGLE']
# box: GLRLM_LRLGLE & stage
y0 = y.loc[x==0]
y1 = y.loc[x==1]
y2 = y.loc[x==2]
y3 = y.loc[x==3]
y4 = y.loc[x==4]
y_box = [y0,y1,y2,y3,y4]

plt.figure(2)
plt.boxplot(y_box, labels=label)
plt.grid(axis='y')
plt.xlabel('Stage')
plt.ylabel('GLRLM_LRLGLE')
plt.show()

# box: GLRLM_LRLGLE & stage(R34)
y0 = y.loc[(x==0)|(x==1)|(x==2)]
y1 = y.loc[(x!=0)&(x!=1)&(x!=2)]
y_box = [y0,y1]

plt.figure(2)
plt.boxplot(y_box, labels=['stage0-2','stage3-4'])
plt.grid(axis='y')
plt.xlabel('Stage')
plt.ylabel('GLRLM_LRLGLE')
plt.show()

#####################################
########     GLRLM_SRLGLE
#####################################
y = data['GLRLM_SRLGLE']
# box: GLRLM_SRLGLE & stage
y0 = y.loc[x==0]
y1 = y.loc[x==1]
y2 = y.loc[x==2]
y3 = y.loc[x==3]
y4 = y.loc[x==4]
y_box = [y0,y1,y2,y3,y4]

plt.figure(2)
plt.boxplot(y_box, labels=label)
plt.grid(axis='y')
plt.xlabel('Stage')
plt.ylabel('GLRLM_SRLGLE')
plt.show()

# box: GLRLM_SRLGLE & stage(R24)
y0 = y.loc[(x==0)|(x==1)]
y1 = y.loc[(x!=0)&(x!=1)]
y_box = [y0,y1]

plt.figure(2)
plt.boxplot(y_box, labels=['stage0-1','stage2-4'])
plt.grid(axis='y')
plt.xlabel('Stage')
plt.ylabel('GLRLM_SRLGLE')
plt.show()