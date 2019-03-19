'''
    Self implement version of simple fully connected NN
    Input Layer = 1 with Size of features
    Output Layer = 1 (Probability to be positive class)
    Hidden Layer = 1 with Size of features * 2
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing

#################################
#### NN define class
#################################
class ANN:
    def __init__(self, Inputsize, Hiddensize, Outputsize, Batchsize, Learning_rate):
        self.Inputsize = Inputsize
        self.Hiddensize = Hiddensize
        self.Outputsize = Outputsize
        self.Batchsize = Batchsize
        self.Learning_rate = Learning_rate
        
        #weight 
        self.w1 = np.random.randn(self.Inputsize, self.Hiddensize)   # input * hidden array
        self.b1 = np.random.randn(1, self.Hiddensize)
        self.w2 = np.random.randn(self.Hiddensize, self.Outputsize)  # hidden * output array
        self.b2 = np.random.randn(1, self.Outputsize)

    def forward(self, X):
        self.z1 = np.dot(X, self.w1) + self.b1
        self.z2 = self.sigmoid(self.z1)
        self.z3 = np.dot(self.z1, self.w2) + self.b2
        output = self.sigmoid(self.z3)
        return output

    def sigmoid(self, input):
        return 1/(1+np.exp(-input))

    def D_sigmoid(self, input):
        return input * (1-input)

    def loss(self, output, Y):
        # incase divide by error warning
        tmp1 = np.maximum(output, 1e-17)
        tmp2 = np.maximum(1-output, 1e-17)
        z1 = Y * np.log(tmp1) + (1-Y) * np.log(tmp2)
        return -np.sum(z1)

    def backward(self, X, Y, output):
        o_loss = self.loss(output, Y)

        D_loss_w2 = np.dot(self.z2.T, (output-Y))
        D_loss_b2 = np.dot(np.ones([1,self.Batchsize]), (output-Y))

        error_through = np.dot((output-Y), self.w2.T) * self.D_sigmoid(self.z2)
        D_loss_w1 = np.dot(X.T, error_through)
        D_loss_b1 = np.dot(np.ones([1,self.Batchsize]), error_through)

        #renew parameters
        # devide (self.Batchsize) to get average gradient
        self.w1 = self.w1 - self.Learning_rate * (D_loss_w1 / self.Batchsize)
        self.b1 = self.b1 - self.Learning_rate * (D_loss_b1 / self.Batchsize)
        self.w2 = self.w2 - self.Learning_rate * (D_loss_w2 / self.Batchsize)
        self.b2 = self.b2 - self.Learning_rate * (D_loss_b2 / self.Batchsize)

        return o_loss        

    def train(self, X, Y, maxiter):
        total_loss = np.zeros([maxiter])

        for iteration in range(maxiter):
            output = self.forward(X)
            Loss = self.backward(X, Y, output)
            total_loss[iteration] = Loss
            if (iteration % 100 == 0):
                print("iteration {:3d} Loss = {:5f}".format(iteration, Loss))

        plt.figure(1)
        plt.plot(total_loss)
        plt.show()

# read in the data
data = pd.read_csv('RFI_TOTAL_train.csv')
y = data['STAGE']
x = data.drop(['PID', 'STAGE', 'SliceNum', 'AREA', 'NLE'], axis=1)
#x = data[['GLRLM_LRLGLE','GLRLM_SRLGLE','GLRLM_LGRE','GLCM_IMC1']]


#####################################
#######      change label
#####################################
R24 = (y > 1).astype(int) # 46 samples
R34 = (y > 2).astype(int) # 21 samples
R4 =  (y > 3).astype(int) # 7 samples

#adjust y 
label = np.array(R34).reshape(-1,1)
# Standardize the feartures
std_x = preprocessing.scale(x, axis= 1)

# split data
x_train, x_test, y_train, y_test = train_test_split(std_x, label, test_size=0.25, random_state=1)

# init NN
NN = ANN(x_train.shape[1], 5, 1, x_train.shape[0], 0.001)
NN.train(x_train, y_train, 1000)

y_prob = NN.forward(x_test)
print("PROB = ",y_prob)
y_pred = (y_prob > 0.5).astype(int)

# ROC, AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(2)
plt.plot(fpr, tpr, lw=1, label='ROC (area = {:0.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], '--', color='r', label='Reference')
plt.legend(loc='upper left')
plt.title('ROC of RFI(stage 0-2 vs 3-4)')
plt.show()

# find best threshold
specificity = 1 - fpr
highest = 0
index = 0
for i in range(fpr.shape[0]):
    ADD = specificity[i] + tpr[i]
    if (ADD > highest):
        highest = ADD
        index = i
TP = (y_prob[y_test==1]) >= thresholds[index] 
TN = (y_prob[y_test==0]) < thresholds[index] 
Accuracy = (sum(TP.astype(int)) + sum(TN.astype(int))) / y_test.shape[0]
print("Threshold:{:8.5f}".format(thresholds[index]))
print("Sensitivity: {:8.2f}%".format(tpr[index]*100))
print("Specificity: {:8.2f}%".format(specificity[index]*100))
# Accuracy 
print("Origin Accuracy(TH = 0.5): %f" % accuracy_score(y_test, y_pred))
print("Accuracy:  {:8.2f}%".format(Accuracy*100))

