import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale, scale

# read in the data
data = pd.read_excel('RFIavg_oneslice.xlsx')
PID = data['PID']
PatientNum = PID.shape[0]
NLE = data['NLE']
STAGE = data['STAGE']
data = data.drop(['PID', 'STAGE', 'SliceNum', 'AREA', 'NLE', 'RFI_Avg'], axis=1)


id = 'BE46'
data1 = pd.read_excel('../Data/' + id + '_Features.xlsx')
y = data1['STAGE']
x = data1.drop(['PID', 'STAGE', 'SLICE', 'AREA', 'RFI_AVG'], axis=1)
x_labels = x.columns
patient_stage = data1['STAGE'].iloc[0]
x = np.array(x)
data = np.array(data)
x = np.array(minmax_scale(x, axis=0))
data = np.array(minmax_scale(data, axis=0))

STD1 = []
STD2 = []
MEAN1 = []
MEAN2 = []
data = data[STAGE == patient_stage]
x_max = np.absolute(x).max(axis=0)
data_max = np.absolute(data).max(axis=0)
for i in range(x.shape[1]):
    STD1 = STD1 + [data[:,i].std()]
    STD2 = STD2 + [x[:,i].std()]
    MEAN1 = MEAN1 + [data[:,i].mean()]
    MEAN2 = MEAN2 + [x[:,i].mean()]

R1 = np.array(STD1) - np.array(STD2)
R2 = np.array(MEAN1) - np.array(MEAN2)
# R1= R1/data_max
# R2 = R2/data_max
plt.figure(1)
plt.scatter(range(x.shape[1]),R1,c='r', label='STD')
plt.scatter(range(x.shape[1]),R2,c='b', label='MEAN')
plt.plot([0,x.shape[1]],[0,0], '--', color='r', label='Reference')

plt.title('STD and MEAN difference of each features(scaled according to max value)')
plt.xticks(range(x.shape[1]),x_labels, rotation='vertical', fontsize=5)
plt.legend(loc='upper right')
plt.ylim([-0.5,0.5])
plt.show()

