import os 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold

kf = GroupKFold(n_splits=5)

# validPID = ['EK46', 'GL43', 'KH32', 'RMT48', 'BG45']  #manually select the valid data incase skew in valid set

# Data = pd.read_csv(os.path.join('..','Label_train_withPID.csv'))
# PID = Data['PID'].unique()

# Valid = pd.DataFrame()

# # trainPID, validPID = train_test_split(PID, test_size=0.25, random_state=1)

# for index in range(len(validPID)):
#     tmp = Data[Data['PID'] == validPID[index]]
#     Valid = Valid.append(tmp)
#     Data = Data[Data['PID'] != validPID[index]]

# Valid = Valid.drop(['PID'], axis= 1)
# Data = Data.drop(['PID'], axis= 1)

# Data.to_csv('../Label_train.csv',index= False)
# Valid.to_csv('../Label_valid.csv',index= False)

# #############################################################################################

# Data = pd.read_csv(os.path.join('..','MaskedLabel_train_withPID.csv'))
# PID = Data['PID'].unique()

# Valid = pd.DataFrame()

# # trainPID, validPID = train_test_split(PID, test_size=0.25, random_state=1)

# for index in range(len(validPID)):
#     tmp = Data[Data['PID'] == validPID[index]]
#     Valid = Valid.append(tmp)
#     Data = Data[Data['PID'] != validPID[index]]

# Valid = Valid.drop(['PID'], axis= 1)
# Data = Data.drop(['PID'], axis= 1)

# Data.to_csv('../MaskedLabel_train.csv',index= False)
# Valid.to_csv('../MaskedLabel_valid.csv',index= False)


#############################################################################################
#random split train valid patient not mutually exclusive
# from sklearn.model_selection import KFold

# kf = KFold(n_splits=5, shuffle=True)
# Data = pd.read_csv(os.path.join('..','MaskedLabel_train_all.csv'))

# for index, (train_index, valid_index) in enumerate(kf.split(Data)):
#     if index==1:
#         Valid = Data.loc[valid_index]
#         Train = Data.loc[train_index]
#         Valid.to_csv(os.path.join('..','MaskedLabel_valid_test.csv'), index=False)
#         Train.to_csv(os.path.join('..','MaskedLabel_train_test.csv'), index=False)
#############################################################################################
# Select area > 2500
Train = pd.read_csv(os.path.join('other_model','RFI_TOTAL_train.csv'))
Train = Train.loc[:,['PID','STAGE','SLICE','AREA']]

data = pd.DataFrame()
data['FileName'] = Train['PID']+'_'+Train['SLICE'].map(str) + '_image.jpg'
data['stage'] = Train['STAGE']
data['PID'] = Train['PID']
data['AREA'] = Train['AREA']
data.to_csv(os.path.join('..','Label_train_all.csv'), index=False)

data = pd.DataFrame()
data['FileName'] = Train['PID']+'_'+Train['SLICE'].map(str) + '_maskedimage.jpg'
data['stage'] = Train['STAGE']
data['PID'] = Train['PID']
data['AREA'] = Train['AREA']
data.to_csv(os.path.join('..','MaskedLabel_train_all.csv'), index=False)

MaskedData = pd.read_csv(os.path.join('..','MaskedLabel_train_all.csv'))
Data = pd.read_csv(os.path.join('..','Label_train_all.csv'))

PID = Data['PID']
# Data = Data.drop(['PID'], axis=1)
# MaskedData = MaskedData.drop(['PID'], axis=1)
LABEL = (Data['stage']>2).astype(int)

for index, (train_index, valid_index) in enumerate(kf.split(Data, LABEL, PID)):
    if index==1:
        MaskedValid = MaskedData.loc[valid_index]
        MaskedTrain = MaskedData.loc[train_index]
        Valid = Data.loc[valid_index]
        Train = Data.loc[train_index]
        MaskedValid.to_csv(os.path.join('..','MaskedLabel_valid_area2500.csv'), index=False)
        MaskedTrain.to_csv(os.path.join('..','MaskedLabel_train_area2500.csv'), index=False)
        Valid.to_csv(os.path.join('..','Label_valid_area2500.csv'), index=False)
        Train.to_csv(os.path.join('..','Label_train_area2500.csv'), index=False)

#############################################################################################
# Select area > 10000
# Masked Train/Valid
Data = pd.read_csv(os.path.join('..','MaskedLabel_train_area2500.csv'))
Data = Data.loc[Data ['AREA']>10000]
Data = Data.drop(['AREA'], axis=1)
Data.to_csv(os.path.join('..','MaskedLabel_train_area10000.csv'), index=False)

Data = pd.read_csv(os.path.join('..','MaskedLabel_valid_area2500.csv'))
Data = Data.loc[Data ['AREA']>10000]
Data = Data.drop(['AREA'], axis=1)
Data.to_csv(os.path.join('..','MaskedLabel_valid_area10000.csv'), index=False)

# Normal Train/Valid
Data = pd.read_csv(os.path.join('..','Label_train_area2500.csv'))
Data = Data.loc[Data ['AREA']>10000]
Data = Data.drop(['AREA'], axis=1)
Data.to_csv(os.path.join('..','Label_train_area10000.csv'), index=False)

Data = pd.read_csv(os.path.join('..','Label_valid_area2500.csv'))
Data = Data.loc[Data ['AREA']>10000]
Data = Data.drop(['AREA'], axis=1)
Data.to_csv(os.path.join('..','Label_valid_area10000.csv'), index=False)

# Select area > 5000
# Masked Train/Valid
Data = pd.read_csv(os.path.join('..','MaskedLabel_train_area2500.csv'))
Data = Data.loc[Data ['AREA']>5000]
Data = Data.drop(['AREA'], axis=1)
Data.to_csv(os.path.join('..','MaskedLabel_train_area5000.csv'), index=False)

Data = pd.read_csv(os.path.join('..','MaskedLabel_valid_area2500.csv'))
Data = Data.loc[Data ['AREA']>5000]
Data = Data.drop(['AREA'], axis=1)
Data.to_csv(os.path.join('..','MaskedLabel_valid_area5000.csv'), index=False)

# Normal Train/Valid
Data = pd.read_csv(os.path.join('..','Label_train_area2500.csv'))
Data = Data.loc[Data ['AREA']>5000]
Data = Data.drop(['AREA'], axis=1)
Data.to_csv(os.path.join('..','Label_train_area5000.csv'), index=False)

Data = pd.read_csv(os.path.join('..','Label_valid_area2500.csv'))
Data  = Data .loc[Data ['AREA']>5000]
Data = Data.drop(['AREA'], axis=1)
Data.to_csv(os.path.join('..','Label_valid_area5000.csv'), index=False)


##################################################################################
#Test
Train = pd.read_csv(os.path.join('other_model','RFI_TOTAL_test.csv'))
Train = Train.loc[:,['PID','STAGE','SLICE','AREA']]

data = pd.DataFrame()
data['FileName'] = Train['PID']+'_'+Train['SLICE'].map(str) + '_image.jpg'
data['stage'] = Train['STAGE']
data['PID'] = Train['PID']
data['AREA'] = Train['AREA']
data.to_csv(os.path.join('..','Label_test_area2500.csv'), index=False)

data = pd.DataFrame()
data['FileName'] = Train['PID']+'_'+Train['SLICE'].map(str) + '_maskedimage.jpg'
data['stage'] = Train['STAGE']
data['PID'] = Train['PID']
data['AREA'] = Train['AREA']
data.to_csv(os.path.join('..','MaskedLabel_test_area2500.csv'), index=False)

# Select area > 5000
Data = pd.read_csv(os.path.join('..','Label_test_area2500.csv'))
Data = Data.loc[Data ['AREA']>5000]
Data = Data.drop(['AREA'], axis=1)
Data.to_csv(os.path.join('..','Label_test_area5000.csv'), index=False)
# Select area > 10000
Data = pd.read_csv(os.path.join('..','Label_test_area2500.csv'))
Data = Data.loc[Data ['AREA']>10000]
Data = Data.drop(['AREA'], axis=1)
Data.to_csv(os.path.join('..','Label_test_area10000.csv'), index=False)

# Select area > 5000
Data = pd.read_csv(os.path.join('..','MaskedLabel_test_area2500.csv'))
Data = Data.loc[Data ['AREA']>5000]
Data = Data.drop(['AREA'], axis=1)
Data.to_csv(os.path.join('..','MaskedLabel_test_area5000.csv'), index=False)
# Select area > 10000
Data = pd.read_csv(os.path.join('..','MaskedLabel_test_area2500.csv'))
Data = Data.loc[Data ['AREA']>10000]
Data = Data.drop(['AREA'], axis=1)
Data.to_csv(os.path.join('..','MaskedLabel_test_area10000.csv'), index=False)



##################################################################################
#HIFI
Train = pd.read_csv(os.path.join('other_model','HIFI.csv'))
Train = Train.loc[:,['PID','STAGE','SLICE','AREA']]

data = pd.DataFrame()
data['FileName'] = Train['PID']+'_'+Train['SLICE'].map(str) + '_image.jpg'
data['stage'] = Train['STAGE']
data['PID'] = Train['PID']
data['AREA'] = Train['AREA']
data.to_csv(os.path.join('..','HIFI_area2500.csv'), index=False)

data = pd.DataFrame()
data['FileName'] = Train['PID']+'_'+Train['SLICE'].map(str) + '_maskedimage.jpg'
data['stage'] = Train['STAGE']
data['PID'] = Train['PID']
data['AREA'] = Train['AREA']
data.to_csv(os.path.join('..','MaskedHIFI_area2500.csv'), index=False)

# Select area > 5000
Data = pd.read_csv(os.path.join('..','HIFI_area2500.csv'))
Data = Data.loc[Data ['AREA']>5000]
Data = Data.drop(['AREA'], axis=1)
Data.to_csv(os.path.join('..','HIFI_area5000.csv'), index=False)
# Select area > 10000
Data = pd.read_csv(os.path.join('..','HIFI_area2500.csv'))
Data = Data.loc[Data ['AREA']>10000]
Data = Data.drop(['AREA'], axis=1)
Data.to_csv(os.path.join('..','HIFI_area10000.csv'), index=False)

# Select area > 5000
Data = pd.read_csv(os.path.join('..','MaskedHIFI_area2500.csv'))
Data = Data.loc[Data ['AREA']>5000]
Data = Data.drop(['AREA'], axis=1)
Data.to_csv(os.path.join('..','MaskedHIFI_area5000.csv'), index=False)
# Select area > 10000
Data = pd.read_csv(os.path.join('..','MaskedHIFI_area2500.csv'))
Data = Data.loc[Data ['AREA']>10000]
Data = Data.drop(['AREA'], axis=1)
Data.to_csv(os.path.join('..','MaskedHIFI_area10000.csv'), index=False)

#########################################################################
# All train data
Data = pd.read_csv(os.path.join('..','MaskedLabel_train_all.csv'))
Data = Data.loc[Data ['AREA']>10000]
Data = Data.drop(['AREA'], axis=1)
Data.to_csv(os.path.join('..','MaskedLabel_train_all_area10000.csv'), index=False)