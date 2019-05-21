import os
import time
import torch
import pandas as pd
from PIL import Image
from argparse import ArgumentParser
# from Resnet import resnet18_pretrain as MODEL
from Resnet import resnet50_pretrain as MODEL
# from Resnet import resnet101_pretrain as MODEL
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score

path = os.path.join('log_Resnet18', 'ResNet50_ImageSize_224_MaskedImage_area10000_aug2(1e-5)_weightdecay(1)', 'checkpoint30.pth')

param = torch.load(path, map_location='cpu')
model = MODEL(pretrain=False)
model.load_state_dict(param)
model.to('cpu')
model.eval()

# path_to_testdata = os.path.join('..','MaskedImage_HIFI')
path_to_testdata = os.path.join('..','MaskedImage')
path_to_testlabel = '../MaskedLabel_valid_area10000.csv'
# path_to_testlabel = '../MaskedLabel_test_area10000.csv'
# path_to_testlabel = '../MaskedHIFI_area10000.csv'

test = pd.read_csv(path_to_testlabel)
Transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

logit = torch.zeros(0)
label = torch.zeros(0)
with torch.no_grad():
    for index in range(len(test)):
        # start = time.time()
        data = test.iloc[index]
        Image_name = data['FileName']
        Stage = data['stage'] 
        label_test = torch.tensor(int(data['stage'] > 2), dtype=torch.float)

        image = Image.open(os.path.join(path_to_testdata,str(Stage),Image_name)).convert('RGB')
        image = Transform(image) 

        logit_test = model(image.view(1,3,224,224))
        # end = time.time()
        # print('Time: ', end-start)

        logit = torch.cat((logit, logit_test.view(1)), dim=0)
        label = torch.cat((label, label_test.view(1)), dim=0)

logit = logit.numpy()
label = label.numpy()

test['LOGIT'] = logit
test.to_csv(os.path.join('Result_valid', 'ResNet50_checkpoint30.csv'), index=False)

plt.figure(1)
plt.scatter(label,logit)
plt.show()

fpr, tpr, thresholds = roc_curve(label,logit)
roc_auc = auc(fpr, tpr)

specificity = 1 - fpr
highest = 0
index = 0
for i in range(fpr.shape[0]):
	ADD = specificity[i] + tpr[i]
	if (ADD > highest):
		highest = ADD
		index = i

plt.figure(2)
plt.plot(fpr, tpr, lw=1, label='ROC (area = {:0.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], '--', color='r', label='Reference')
plt.legend(loc='upper left')
plt.title('ROC of ResNet(stage 0-2 vs 3-4)')
plt.show()

totalP = label[label==1].size
totalN = label[label==0].size
logit = (logit>0.5).astype(int)

print(logit[label==1].sum(), '   ', totalP)
print(totalN-logit[label==0].sum(), '   ', totalN)
TP = logit[label==1].sum()/totalP
TN = 1 - logit[label==0].sum()/totalN
print('Threshold: ', thresholds[index])
print('Sensitivity: ', TP)
print('Specificity: ', TN)
print('Accuracy   : ', accuracy_score(logit,label))