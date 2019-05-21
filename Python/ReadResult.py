import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score

# onarea2500_aug1 = os.path.join('.', 'Result_valid', 'Masked_area2500_on_area2500_aug1.csv')
# onarea5000_aug1 = os.path.join('.', 'Result_valid', 'Masked_area2500_on_area5000_aug1.csv')
# onarea10000_aug1 = os.path.join('.', 'Result_valid', 'Masked_area2500_on_area10000_aug1.csv')
# onarea2500_aug2 = os.path.join('.', 'Result_valid', 'Masked_area2500_on_area2500_aug2(1e-5).csv')
# onarea5000_aug2 = os.path.join('.', 'Result_valid', 'Masked_area2500_on_area5000_aug2(1e-5).csv')
# onarea10000_aug2 = os.path.join('.', 'Result_valid', 'Masked_area2500_on_area10000_aug2(1e-5).csv')
# name = ['onarea2500_aug1', 'onarea5000_aug1', 'onarea10000_aug1', 'onarea2500_aug2', 'onarea5000_aug2','onarea10000_aug2']
# path = [onarea2500_aug1, onarea5000_aug1 , onarea10000_aug1, onarea2500_aug2, onarea5000_aug2, onarea10000_aug2]

# area10000_noweightdecay = os.path.join('.', 'Result_valid', 'Masked_area10000_on_area10000_aug2(1e-5).csv')
# area10000_weightdecay_1 = os.path.join('.', 'Result_valid', 'Masked_area10000_on_area10000_aug2(1e-5)weightdecay(0.1).csv') 
# area10000_weightdecay_2 = os.path.join('.', 'Result_valid', 'Masked_area10000_on_area10000_aug2(1e-5)weightdecay(1).csv') 
# name = ['noweightdecay', 'weightdecay(0.1)', 'weightdecay(1)']
# path = [area10000_noweightdecay, area10000_weightdecay_1 , area10000_weightdecay_2]

# area10000onarea10000_checkpoint10 = os.path.join('.', 'Result_valid', 'Masked_area10000_on_area10000_aug2(1e-5)weightdecay(1)_checkpoint10.csv') 
# area10000onarea10000_checkpoint15 = os.path.join('.', 'Result_valid', 'Masked_area10000_on_area10000_aug2(1e-5)weightdecay(1)_checkpoint15.csv') 
# area10000onarea10000_checkpoint20 = os.path.join('.', 'Result_valid', 'Masked_area10000_on_area10000_aug2(1e-5)weightdecay(1)_checkpoint20.csv') 
# area10000onarea10000_checkpoint25 = os.path.join('.', 'Result_valid', 'Masked_area10000_on_area10000_aug2(1e-5)weightdecay(1)_checkpoint25.csv') 
# area10000onarea10000_checkpoint30 = os.path.join('.', 'Result_valid', 'Masked_area10000_on_area10000_aug2(1e-5)weightdecay(1).csv') 
# area10000onarea10000aug1_checkpoint10 = os.path.join('.', 'Result_valid', 'Masked_area10000_on_area10000_aug1(1e-5)weightdecay(1)_checkpoint10.csv') 
# area10000onarea10000aug1_checkpoint15 = os.path.join('.', 'Result_valid', 'Masked_area10000_on_area10000_aug1(1e-5)weightdecay(1)_checkpoint15.csv') 
# area10000onarea10000aug1_checkpoint20 = os.path.join('.', 'Result_valid', 'Masked_area10000_on_area10000_aug1(1e-5)weightdecay(1)_checkpoint20.csv') 
# area10000onarea10000aug1_checkpoint25 = os.path.join('.', 'Result_valid', 'Masked_area10000_on_area10000_aug1(1e-5)weightdecay(1)_checkpoint25.csv') 
# area10000onarea10000aug1_checkpoint30 = os.path.join('.', 'Result_valid', 'Masked_area10000_on_area10000_aug1(1e-5)weightdecay(1).csv') 
# name = ['checkpoint10', 'checkpoint15', 'checkpoint20', 'checkpoint25', 'checkpoint30',
#         'checkpoint10aug1', 'checkpoint15aug1', 'checkpoint20aug1', 'checkpoint25aug1', 'checkpoint30aug1']
# path = [area10000onarea10000_checkpoint10, area10000onarea10000_checkpoint15 , area10000onarea10000_checkpoint20, area10000onarea10000_checkpoint25, area10000onarea10000_checkpoint30,
#         area10000onarea10000aug1_checkpoint10, area10000onarea10000aug1_checkpoint15 , area10000onarea10000aug1_checkpoint20, area10000onarea10000aug1_checkpoint25, area10000onarea10000aug1_checkpoint30]
# name = ['checkpoint10', 'checkpoint15', 'checkpoint20', 'checkpoint25', 'checkpoint30']
# path = [area10000onarea10000_checkpoint10, area10000onarea10000_checkpoint15 , area10000onarea10000_checkpoint20, area10000onarea10000_checkpoint25, area10000onarea10000_checkpoint30]
# name = ['checkpoint10', 'checkpoint15', 'checkpoint20', 'checkpoint25', 'checkpoint30']
# path = [area10000onarea10000aug1_checkpoint10, area10000onarea10000aug1_checkpoint15 , area10000onarea10000aug1_checkpoint20, area10000onarea10000aug1_checkpoint25, area10000onarea10000aug1_checkpoint30]


# area2500_checkpoint10 = os.path.join('.', 'Result_valid', 'Masked_area2500_on_area2500_aug2(1e-5)weightdecay(1)_checkpoint10.csv') 
# area2500_checkpoint15 = os.path.join('.', 'Result_valid', 'Masked_area2500_on_area2500_aug2(1e-5)weightdecay(1)_checkpoint15.csv') 
# area2500_checkpoint20 = os.path.join('.', 'Result_valid', 'Masked_area2500_on_area2500_aug2(1e-5)weightdecay(1)_checkpoint20.csv') 
# area2500_checkpoint25 = os.path.join('.', 'Result_valid', 'Masked_area2500_on_area2500_aug2(1e-5)weightdecay(1)_checkpoint25.csv') 
# area2500_checkpoint30 = os.path.join('.', 'Result_valid', 'Masked_area2500_on_area2500_aug2(1e-5)weightdecay(1).csv') 
# area5000_checkpoint10 = os.path.join('.', 'Result_valid', 'Masked_area2500_on_area5000_aug2(1e-5)weightdecay(1)_checkpoint10.csv') 
# area5000_checkpoint15 = os.path.join('.', 'Result_valid', 'Masked_area2500_on_area5000_aug2(1e-5)weightdecay(1)_checkpoint15.csv') 
# area5000_checkpoint20 = os.path.join('.', 'Result_valid', 'Masked_area2500_on_area5000_aug2(1e-5)weightdecay(1)_checkpoint20.csv') 
# area5000_checkpoint25 = os.path.join('.', 'Result_valid', 'Masked_area2500_on_area5000_aug2(1e-5)weightdecay(1)_checkpoint25.csv') 
# area5000_checkpoint30 = os.path.join('.', 'Result_valid', 'Masked_area2500_on_area5000_aug2(1e-5)weightdecay(1).csv') 
# area10000_checkpoint10 = os.path.join('.', 'Result_valid', 'Masked_area2500_on_area10000_aug2(1e-5)weightdecay(1)_checkpoint10.csv') 
# area10000_checkpoint15 = os.path.join('.', 'Result_valid', 'Masked_area2500_on_area10000_aug2(1e-5)weightdecay(1)_checkpoint15.csv') 
# area10000_checkpoint20 = os.path.join('.', 'Result_valid', 'Masked_area2500_on_area10000_aug2(1e-5)weightdecay(1)_checkpoint20.csv') 
# area10000_checkpoint25 = os.path.join('.', 'Result_valid', 'Masked_area2500_on_area10000_aug2(1e-5)weightdecay(1)_checkpoint25.csv') 
# area10000_checkpoint30 = os.path.join('.', 'Result_valid', 'Masked_area2500_on_area10000_aug2(1e-5)weightdecay(1).csv') 
# name = ['area10000_checkpoint10', 'area10000_checkpoint15', 'area10000_checkpoint20','area10000_checkpoint25', 'area10000_checkpoint30',
#         'area5000_checkpoint10', 'area5000_checkpoint15', 'area5000_checkpoint20','area5000_checkpoint25', 'area5000_checkpoint30',
#         'area2500_checkpoint10', 'area2500_checkpoint15', 'area2500_checkpoint20','area2500_checkpoint25', 'area2500_checkpoint30']
# path = [area10000_checkpoint10, area10000_checkpoint15  , area10000_checkpoint20, area10000_checkpoint25, area10000_checkpoint25,
#         area5000_checkpoint10, area5000_checkpoint15  , area5000_checkpoint20, area5000_checkpoint25, area5000_checkpoint30,
#         area2500_checkpoint10, area2500_checkpoint15  , area2500_checkpoint20, area2500_checkpoint25, area2500_checkpoint30]


# checkpoint10 = os.path.join('.', 'Result', 'Masked_area10000_on_area10000_aug2(1e-5)weightdecay(1)_checkpoint10.csv')
# checkpoint15 = os.path.join('.', 'Result', 'Masked_area10000_on_area10000_aug2(1e-5)weightdecay(1)_checkpoint15.csv')
# checkpoint20 = os.path.join('.', 'Result', 'Masked_area10000_on_area10000_aug2(1e-5)weightdecay(1)_checkpoint20.csv')
# checkpoint25 = os.path.join('.', 'Result', 'Masked_area10000_on_area10000_aug2(1e-5)weightdecay(1)_checkpoint25.csv')
# checkpoint30 = os.path.join('.', 'Result', 'Masked_area10000_on_area10000_aug2(1e-5)weightdecay(1).csv')
# name = ['checkpoint10', 'checkpoint15','checkpoint20', 'checkpoint25', 'checkpoint30']
# path = [checkpoint10, checkpoint15, checkpoint20, checkpoint25, checkpoint30]

# checkpoint10aug1 = os.path.join('.', 'Result', 'Masked_area10000_on_area10000_aug1(1e-5)weightdecay(1)_checkpoint10.csv')
# checkpoint15aug1 = os.path.join('.', 'Result', 'Masked_area10000_on_area10000_aug1(1e-5)weightdecay(1)_checkpoint15.csv')
# checkpoint20aug1 = os.path.join('.', 'Result', 'Masked_area10000_on_area10000_aug1(1e-5)weightdecay(1)_checkpoint20.csv')
# checkpoint30aug1 = os.path.join('.', 'Result', 'Masked_area10000_on_area10000_aug1(1e-5)weightdecay(1).csv')
# name = ['checkpoint10', 'checkpoint15','checkpoint20', 'checkpoint30']
# path = [checkpoint10aug1, checkpoint15aug1, checkpoint20aug1, checkpoint30aug1]

# Inception_checkpoint20 = os.path.join('.', 'Result_HIFI', 'InceptionV3_area10000_aug2(1e-5)weightdecay(1)_checkpoint20.csv')
# ResNet101_checkpoint20 = os.path.join('.', 'Result_HIFI', 'ResNet101_area10000_aug2(1e-5)weightdecay(1)_checkpoint20.csv')
# ResNet50_checkpoint20 = os.path.join('.', 'Result_HIFI', 'ResNet50_area10000_aug2(1e-5)weightdecay(1)_checkpoint20.csv')
# ResNet18_checkpoint15 = os.path.join('.', 'Result_HIFI', 'Masked_area10000_on_area10000_aug2(1e-5)weightdecay(1)_checkpoint15.csv')
# ResNet18_checkpoint20 = os.path.join('.', 'Result_HIFI', 'Masked_area10000_on_area10000_aug2(1e-5)weightdecay(1)_checkpoint20.csv')
# ResNet18_notrain = os.path.join('.', 'Result_HIFI', 'ResNet18_notrain.csv')
# ResNet18_pretrain = os.path.join('.', 'Result_HIFI', 'ResNet18_pretrain.csv')
# name = ['InceptionV3', 'ResNet101', 'ResNet50','ResNet18_checkpoint15', 'ResNet18_checkpoint20', 'ResNet18_notrain', 'ResNet18_pretrain']
# path = [Inception_checkpoint20, ResNet101_checkpoint20, ResNet50_checkpoint20, ResNet18_checkpoint15, ResNet18_checkpoint20, ResNet18_notrain, ResNet18_pretrain]

# Inception_checkpoint15 = os.path.join('.', 'Result', 'InceptionV3_area10000_aug2(1e-5)weightdecay(1)_checkpoint15.csv')
# Inception_checkpoint20 = os.path.join('.', 'Result', 'InceptionV3_area10000_aug2(1e-5)weightdecay(1)_checkpoint20.csv')
# Inception_checkpoint30 = os.path.join('.', 'Result', 'InceptionV3_area10000_aug2(1e-5)weightdecay(1).csv')
# ResNet101_checkpoint15 = os.path.join('.', 'Result', 'ResNet101_area10000_aug2(1e-5)weightdecay(1)_checkpoint15.csv')
# ResNet101_checkpoint20 = os.path.join('.', 'Result', 'ResNet101_area10000_aug2(1e-5)weightdecay(1)_checkpoint20.csv')
# ResNet101_checkpoint30 = os.path.join('.', 'Result', 'ResNet101_area10000_aug2(1e-5)weightdecay(1)_checkpoint30.csv')
# ResNet50_checkpoint15 = os.path.join('.', 'Result', 'ResNet50_area10000_aug2(1e-5)weightdecay(1)_checkpoint15.csv')
# ResNet50_checkpoint20 = os.path.join('.', 'Result', 'ResNet50_area10000_aug2(1e-5)weightdecay(1)_checkpoint20.csv')
# ResNet50_checkpoint30 = os.path.join('.', 'Result', 'ResNet50_area10000_aug2(1e-5)weightdecay(1)_checkpoint30.csv')
# ResNet18_checkpoint15 = os.path.join('.', 'Result', 'Masked_area10000_on_area10000_aug2(1e-5)weightdecay(1)_checkpoint15.csv')
# ResNet18_checkpoint20 = os.path.join('.', 'Result', 'Masked_area10000_on_area10000_aug2(1e-5)weightdecay(1)_checkpoint20.csv')
# ResNet18_checkpoint30 = os.path.join('.', 'Result', 'Masked_area10000_on_area10000_aug2(1e-5)weightdecay(1).csv')
# name = ['Inception_checkpoint15', 'Inception_checkpoint20', 'Inception_checkpoint30',
#         'ResNet101_checkpoint15', 'ResNet101_checkpoint20', 'ResNet101_checkpoint30',
#         'ResNet50_checkpoint15', 'ResNet50_checkpoint20', 'ResNet50_checkpoint30',
#         'ResNet18_checkpoint15', 'ResNet18_checkpoint20', 'ResNet18_checkpoint30']
# path = [Inception_checkpoint15, Inception_checkpoint20, Inception_checkpoint30,
#         ResNet101_checkpoint15, ResNet101_checkpoint20, ResNet101_checkpoint30,
#         ResNet50_checkpoint15, ResNet50_checkpoint20, ResNet50_checkpoint30,
#         ResNet18_checkpoint15, ResNet18_checkpoint20, ResNet18_checkpoint30]

# Inception_checkpoint15 = os.path.join('.', 'Result_valid', 'InceptionV3_checkpoint15.csv')
# Inception_checkpoint20 = os.path.join('.', 'Result_valid', 'InceptionV3_checkpoint20.csv')
# Inception_checkpoint30 = os.path.join('.', 'Result_valid', 'InceptionV3_checkpoint30.csv')
# ResNet101_checkpoint15 = os.path.join('.', 'Result_valid', 'ResNet101_checkpoint15.csv')
# ResNet101_checkpoint20 = os.path.join('.', 'Result_valid', 'ResNet101_checkpoint20.csv')
# ResNet101_checkpoint30 = os.path.join('.', 'Result_valid', 'ResNet101_checkpoint30.csv')
# ResNet50_checkpoint15 = os.path.join('.', 'Result_valid', 'ResNet50_checkpoint15.csv')
# ResNet50_checkpoint20 = os.path.join('.', 'Result_valid', 'ResNet50_checkpoint20.csv')
# ResNet50_checkpoint30 = os.path.join('.', 'Result_valid', 'ResNet50_checkpoint30.csv')
# name = ['Inception_checkpoint15', 'Inception_checkpoint20', 'Inception_checkpoint30',
#         'ResNet101_checkpoint15', 'ResNet101_checkpoint20', 'ResNet101_checkpoint30',
#         'ResNet50_checkpoint15', 'ResNet50_checkpoint20', 'ResNet50_checkpoint30']
# path = [Inception_checkpoint15, Inception_checkpoint20, Inception_checkpoint30,
#         ResNet101_checkpoint15, ResNet101_checkpoint20, ResNet101_checkpoint30,
#         ResNet50_checkpoint15, ResNet50_checkpoint20, ResNet50_checkpoint30]

c = ['b','g', 'r','c','m','y','k','w']
AUC = []
Sensitivity = []
Specifivity = []
Accuracy = []
plt.figure()

for index, p in enumerate(path):
    data=pd.read_csv(p)
    label = (data['stage'] > 2).astype(int)
    logit = data['LOGIT']

    fpr, tpr, thresholds = roc_curve(label,logit)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, lw=1, label=name[index]+'(area = {:0.2f})'.format(roc_auc), c=c[index//3])  

    AUC.append(roc_auc)

    totalP = label[label==1].size
    totalN = label[label==0].size
    logit = (logit>0.5).astype(int)
    TP = logit[label==1].sum()/totalP
    TN = 1 - logit[label==0].sum()/totalN

    Sensitivity.append(TP)
    Specifivity.append(TN)
    Accuracy.append(accuracy_score(logit,label))

plt.plot([0, 1], [0, 1], '--', color='r', label='Reference')
plt.legend(loc='lower right')
plt.title('ROC of stage 0-2 vs 3-4 (train area10000 with aug2 and weightdecay)')
plt.show()

for index in range(9):
    print('Sensitivity: ', Sensitivity[index])
    print('Specificity: ', Specifivity[index])
    print('Accuracy   : ', Accuracy[index])
