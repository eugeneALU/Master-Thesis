import os 
import pandas as pd 
import matplotlib.pyplot as plt 


filepath = ['2_Result_MAXMAX.csv','2_Result_minMAX.csv','2_Result_minmin.csv']

for path in filepath:
    data = pd.read_csv(os.path.join('..', path))

    ONE = data[data['LABEL']==1]['DISTANCE']
    ONEy = data[data['LABEL']==1]['LABEL']

    ZERO = data[data['LABEL']==0]['DISTANCE']
    ZEROy = data[data['LABEL']==0]['LABEL']

    plt.figure()
    plt.scatter(ONEy,ONE)
    plt.scatter(ZEROy,ZERO)
    plt.xticks([0,1])
    plt.ylabel('DISTANCE')
    plt.title(path+' distance distribution')
    plt.show()