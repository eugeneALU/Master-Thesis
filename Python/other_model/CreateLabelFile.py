import os 
import pandas as pd

supfile = ['0','1','2','3','4']
data = pd.DataFrame()

for index in range(5):
    for root, dirs, files in os.walk(os.path.join('..','Image_train',supfile[index])):
        tmp = pd.DataFrame(files[1:]) #deal with .DS_Store
        tmp.columns = ['FileName']
        tmp['stage'] = index
        data = data.append(tmp)
data.to_csv('../Label_train.csv',index= False)

data = pd.DataFrame()

for index in range(5):
    for root, dirs, files in os.walk(os.path.join('..','Image_test',supfile[index])):
        tmp = pd.DataFrame(files[1:]) #deal with .DS_Store
        tmp.columns = ['FileName']
        tmp['stage'] = index
        data = data.append(tmp)
data.to_csv('../Label_test.csv',index= False)
