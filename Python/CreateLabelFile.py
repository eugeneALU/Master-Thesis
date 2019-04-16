import os 
import pandas as pd

supfile = ['0','1','2','3','4']
data = pd.DataFrame()

for index in range(5):
    for root, dirs, files in os.walk(os.path.join('..','MaskedImage_train',supfile[index])):
        tmp = pd.DataFrame(files)
        tmp.columns = ['FileName']
        tmp['stage'] = index
        data = data.append(tmp)
data = data[data.FileName != 'Thumbs.db']
data = data[data.FileName != '.DS_Store']
data.to_csv('../MaskedLabel_train_all.csv',index= False)

data = pd.DataFrame()

for index in range(5):
    for root, dirs, files in os.walk(os.path.join('..','MaskedImage_test',supfile[index])):
        tmp = pd.DataFrame(files)
        tmp.columns = ['FileName']
        tmp['stage'] = index
        data = data.append(tmp)
data = data[data.FileName != 'Thumbs.db']
data = data[data.FileName != '.DS_Store']
data.to_csv('../MaskedLabel_test.csv',index= False)

data = pd.DataFrame()

for index in range(5):
    for root, dirs, files in os.walk(os.path.join('..','Image_train',supfile[index])):
        tmp = pd.DataFrame(files)
        tmp.columns = ['FileName']
        tmp['stage'] = index
        data = data.append(tmp)
data = data[data.FileName != 'Thumbs.db']
data = data[data.FileName != '.DS_Store']
data.to_csv('../Label_train_all.csv',index= False)

data = pd.DataFrame()

for index in range(5):
    for root, dirs, files in os.walk(os.path.join('..','Image_test',supfile[index])):
        tmp = pd.DataFrame(files)
        tmp.columns = ['FileName']
        tmp['stage'] = index
        data = data.append(tmp)
data = data[data.FileName != 'Thumbs.db']
data = data[data.FileName != '.DS_Store']
data.to_csv('../Label_test.csv',index= False)


data = pd.DataFrame()

for index in range(5):
    for root, dirs, files in os.walk(os.path.join('..','Image_train',supfile[index])):
        tmp = pd.DataFrame(files)
        tmp.columns = ['FileName']
        tmp['stage'] = index
        data = data.append(tmp)
data = data[data.FileName != 'Thumbs.db']
data = data[data.FileName != '.DS_Store']
data['PID'] = ''
for index in range(data.shape[0]):
        PID = data['FileName'].iloc[index].split('_',1)[0]
        data['PID'].iloc[index] = PID
data.to_csv('../Label_train_withPID.csv',index= False)

data = pd.DataFrame()

for index in range(5):
    for root, dirs, files in os.walk(os.path.join('..','MaskedImage_train',supfile[index])):
        tmp = pd.DataFrame(files)
        tmp.columns = ['FileName']
        tmp['stage'] = index
        data = data.append(tmp)
data = data[data.FileName != 'Thumbs.db']
data = data[data.FileName != '.DS_Store']
data['PID'] = ''
for index in range(data.shape[0]):
        PID = data['FileName'].iloc[index].split('_',1)[0]
        data['PID'].iloc[index] = PID
data.to_csv('../MaskedLabel_train_withPID.csv',index= False)
