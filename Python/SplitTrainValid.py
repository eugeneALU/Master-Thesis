import os 
import pandas as pd
from sklearn.model_selection import train_test_split

validPID = ['EK46', 'GL43', 'KH32', 'RMT48', 'BG45']  #manually select the valid data incase skew in valid set

Data = pd.read_csv(os.path.join('..','Label_train_withPID.csv'))
PID = Data['PID'].unique()

Valid = pd.DataFrame()

# trainPID, validPID = train_test_split(PID, test_size=0.25, random_state=1)

for index in range(len(validPID)):
    tmp = Data[Data['PID'] == validPID[index]]
    Valid = Valid.append(tmp)
    Data = Data[Data['PID'] != validPID[index]]

Valid = Valid.drop(['PID'], axis= 1)
Data = Data.drop(['PID'], axis= 1)

Data.to_csv('../Label_train.csv',index= False)
Valid.to_csv('../Label_valid.csv',index= False)

#############################################################################################

Data = pd.read_csv(os.path.join('..','MaskedLabel_train_withPID.csv'))
PID = Data['PID'].unique()

Valid = pd.DataFrame()

# trainPID, validPID = train_test_split(PID, test_size=0.25, random_state=1)

for index in range(len(validPID)):
    tmp = Data[Data['PID'] == validPID[index]]
    Valid = Valid.append(tmp)
    Data = Data[Data['PID'] != validPID[index]]

Valid = Valid.drop(['PID'], axis= 1)
Data = Data.drop(['PID'], axis= 1)

Data.to_csv('../MaskedLabel_train.csv',index= False)
Valid.to_csv('../MaskedLabel_valid.csv',index= False)