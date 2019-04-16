import os
import pandas as pd

train_path = os.path.join('other_model','RFIavg_oneslice_train.xlsx')
test_path = os.path.join('other_model','RFIavg_oneslice_test.xlsx')
train_name = 'PairLabel_train.csv'
test_name = 'PairLabel_test.csv'
PATH = [train_path, test_path]
NAME = [train_name, test_name]

for path, name in zip(PATH, NAME):
    # read in the data
    data = pd.read_excel(path)
    PID = data['PID']
    PatientNum = PID.shape[0]
    del data

    DATA = pd.DataFrame()
    tmp = pd.DataFrame()
    for pid,index in zip(PID, range(PatientNum)):
        try:
            data = pd.read_excel('../Data/' + pid + '_Features.xlsx')
            M = data['AREA'].max()                   # select the slice with largest liver area
            maxSlice = data[data['AREA'] == M]
            tmp = tmp.append(maxSlice.loc[:,'PID':'AREA'], ignore_index=True)
        except:
            print("FILE not exist: ", pid)
            continue

    pair = pd.Series()
    for index1 in range(len(tmp)):
        for index2 in range(len(tmp)):
            if index1 < index2: 
                x1 = tmp.iloc[index1]
                x2 = tmp.iloc[index2]

                pair['PID1'] = x1.PID + '_' + str(x1.SLICE)
                pair['PID2'] = x2.PID + '_' + str(x2.SLICE)
                pair['STAGE1'] = x1.STAGE
                pair['STAGE2'] = x2.STAGE
                pair['RFI_AVG1'] = x1.RFI_AVG
                pair['RFI_AVG2'] = x2.RFI_AVG

                # same stage label = 1, otherwise = 0
                if x1.STAGE == x2.STAGE:
                    pair['LABEL'] = 1
                else:
                    pair['LABEL'] = 0

                DATA = DATA.append(pair, ignore_index=True)    

    DATA.to_csv(os.path.join('..',name), index= False)    

