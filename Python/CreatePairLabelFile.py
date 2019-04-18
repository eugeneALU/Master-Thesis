import os
import pandas as pd

train_path = os.path.join('other_model','RFIavg_oneslice_train.xlsx')
test_path = os.path.join('other_model','RFIavg_oneslice_test.xlsx')
train_name = 'PairLabel_train'
test_name = 'PairLabel_test'
PATH = [train_path, test_path]
NAME = [train_name, test_name]

# generate min vs min and max vs min and max vs max
for path, name in zip(PATH, NAME):
    # read in the data
    data = pd.read_excel(path)
    PID = data['PID']
    PatientNum = PID.shape[0]
    del data

    tmpMax = pd.DataFrame()
    tmpMin = pd.DataFrame()
    for pid,index in zip(PID, range(PatientNum)):
        try:
            data = pd.read_excel('../Data/' + pid + '_Features.xlsx')
            maxSlice = data[data['AREA'] == data['AREA'].max()]
            minSlice = data[data['AREA'] == data['AREA'].min()]
            tmpMax = tmpMax.append(maxSlice.loc[:,'PID':'AREA'], ignore_index=True)
            tmpMin = tmpMin.append(minSlice.loc[:,'PID':'AREA'], ignore_index=True)
        except:
            print("FILE not exist: ", pid)
            continue

    pair = pd.Series()
    DATA = pd.DataFrame()
    for index1 in range(len(tmpMax)):
        for index2 in range(len(tmpMax)):
            if index1 < index2: 
                x1 = tmpMax.iloc[index1]
                x2 = tmpMax.iloc[index2]

                pair['PID1'] = x1.PID + '_' + str(x1.SLICE)
                pair['PID2'] = x2.PID + '_' + str(x2.SLICE)
                pair['STAGE1'] = x1.STAGE
                pair['STAGE2'] = x2.STAGE
                # pair['RFI_AVG1'] = x1.RFI_AVG
                # pair['RFI_AVG2'] = x2.RFI_AVG
                pair['AREA1'] = x1.AREA
                pair['AREA2'] = x2.AREA

                # same stage label = 1, otherwise = 0
                if x1.STAGE == x2.STAGE:
                    pair['LABEL'] = 1
                else:
                    pair['LABEL'] = 0

                DATA = DATA.append(pair, ignore_index=True)    

    DATA.to_csv(os.path.join('..', name + '_MAXMAX.csv'), index= False)    

    pair = pd.Series()
    DATA = pd.DataFrame()
    for index1 in range(len(tmpMin)):
        for index2 in range(len(tmpMin)):
            if index1 < index2: 
                x1 = tmpMin.iloc[index1]
                x2 = tmpMin.iloc[index2]

                pair['PID1'] = x1.PID + '_' + str(x1.SLICE)
                pair['PID2'] = x2.PID + '_' + str(x2.SLICE)
                pair['STAGE1'] = x1.STAGE
                pair['STAGE2'] = x2.STAGE
                # pair['RFI_AVG1'] = x1.RFI_AVG
                # pair['RFI_AVG2'] = x2.RFI_AVG
                pair['AREA1'] = x1.AREA
                pair['AREA2'] = x2.AREA

                # same stage label = 1, otherwise = 0
                if x1.STAGE == x2.STAGE:
                    pair['LABEL'] = 1
                else:
                    pair['LABEL'] = 0

                DATA = DATA.append(pair, ignore_index=True)    

    DATA.to_csv(os.path.join('..',name + '_minmin.csv'), index= False)  

    pair = pd.DataFrame()
    DATA = pd.DataFrame()
    for index1 in range(len(tmpMin)):
        for index2 in range(len(tmpMin)):
            if index1 < index2: 
                x1 = tmpMin.iloc[index1]
                x2 = tmpMax.iloc[index1]
                x3 = tmpMin.iloc[index2]
                x4 = tmpMax.iloc[index2]

                d = {
                    'PID1': [x1.PID + '_' + str(x1.SLICE), x2.PID + '_' + str(x2.SLICE)], 
                    'PID2': [x4.PID + '_' + str(x4.SLICE), x3.PID + '_' + str(x3.SLICE)], 
                    'STAGE1':[x1.STAGE, x2.STAGE],
                    'STAGE2':[x4.STAGE, x3.STAGE],
                    'AREA1':[x1.AREA, x2.AREA],
                    'AREA2':[x4.AREA, x3.AREA],
                    'LABEL':[int(x1.STAGE == x4.STAGE), int(x2.STAGE == x3.STAGE)]
                    }
                
                pair = pd.DataFrame(data=d)
                DATA = DATA.append(pair, ignore_index=True)    

    DATA.to_csv(os.path.join('..',name + '_minMAX.csv'), index= False)    

### create train valid file
# Train1 = pd.read_csv(os.path.join('..','PairLabel_train_MAXMAX.csv'))
# Train2 = pd.read_csv(os.path.join('..','PairLabel_train_minMAX.csv'))
# Train3 = pd.read_csv(os.path.join('..','PairLabel_train_minmin.csv'))
# pd.concat([Train1]).to_csv(os.path.join('..', 'PairLabel_train.csv'), index= False)
# pd.concat([Train2, Train3]).to_csv(os.path.join('..', 'PairLabel_valid.csv'), index= False)