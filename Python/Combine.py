import pandas as pd

# read in the data
data = pd.read_excel('Result_train.xlsx')
PID = data['PID']
PatientNum = PID.shape[0]
NLE = data['NLE']
del data

TOTAL = pd.DataFrame()
for id,index in zip(PID, range(PatientNum)):
    try:
        data = pd.read_excel('../Data/' + id + '_Features.xlsx')
        data['NLE'] = NLE[index]
        TOTAL = TOTAL.append(data)
    except:
        continue

TOTAL.to_csv('TOTAL_train.csv', index= False)
del TOTAL

# read in the data
data = pd.read_excel('Result_test.xlsx')
PID = data['PID']
PatientNum = PID.shape[0]
NLE = data['NLE']
del data

TOTAL = pd.DataFrame()
for id,index in zip(PID, range(PatientNum)):
    try:
        data = pd.read_excel('../Data/' + id + '_Features.xlsx')
        data['NLE'] = NLE[index]
        TOTAL = TOTAL.append(data)
    except:
        continue

TOTAL.to_csv('TOTAL_test.csv', index= False)