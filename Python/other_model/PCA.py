import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# read in the data
# data = pd.read_excel('Liver_normalize_Local_quantize_one slice per patient.xlsx')
# x = data.drop(['PID', 'STAGE', 'AREA'], axis=1)
data = pd.read_csv('RFI_TOTAL_train.csv')
# data = data.append(pd.read_csv('RFI_TOTAL_test.csv'))
# data = data.sample(n=1000, replace=False, random_state=1)
y = data['STAGE']
A = data['AREA']
PID = data['PID']
M = A.mean()
x = data.drop(['PID', 'STAGE', 'SLICE', 'AREA', 'NLE'], axis=1)
# y = y[A>M]
# x = x[A>M]

# data = pd.read_excel('RFIavg_oneslice_train.xlsx')
data = pd.read_csv('RFI_TOTAL_test.csv')
y_test = data['STAGE']
# A_test = data['AREA']
# M_test = A_test.mean()
x_test = data.drop(['PID', 'STAGE', 'SLICE', 'AREA', 'NLE'], axis=1)
# y_test = y_test[A_test>M_test]
# x_test = x_test[A_test>M_test]

pca = PCA(n_components=2, svd_solver='full')

pcax = pca.fit_transform(x)
# pcax_test = pca.transform(x_test)


blue_patch = mpatches.Patch(color='#0000FF', label='stage0')
green_patch = mpatches.Patch(color='g', label='stage1')
yellow_patch = mpatches.Patch(color='#FFFF00', label='stage2')
orange_patch = mpatches.Patch(color='#FFBF00', label='stage3')
red_patch = mpatches.Patch(color='red', label='stage4')

plt.figure()
plt.title('data after PCA (to 2D)')
plt.scatter(pcax[y==0][:,0],pcax[y==0][:,1], c='#0000FF')
plt.scatter(pcax[y==1][:,0],pcax[y==1][:,1], c='g')
plt.scatter(pcax[y==2][:,0],pcax[y==2][:,1], c='#FFFF00')
plt.scatter(pcax[y==3][:,0],pcax[y==3][:,1], c='#FFBF00')
plt.scatter(pcax[y==4][:,0],pcax[y==4][:,1], c='r')
plt.legend(handles=[blue_patch,green_patch,yellow_patch,orange_patch,red_patch])
plt.show()

# plt.figure()
# plt.title('data after PCA (to 2D)')
# plt.scatter(pcax_test[y_test==0][:,0],pcax_test[y_test==0][:,1], c='#0000FF')
# plt.scatter(pcax_test[y_test==1][:,0],pcax_test[y_test==1][:,1], c='g')
# plt.scatter(pcax_test[y_test==2][:,0],pcax_test[y_test==2][:,1], c='#FFFF00')
# plt.scatter(pcax_test[y_test==3][:,0],pcax_test[y_test==3][:,1], c='#FFBF00')
# plt.scatter(pcax_test[y_test==4][:,0],pcax_test[y_test==4][:,1], c='r')
# plt.legend(handles=[blue_patch,green_patch,yellow_patch,orange_patch,red_patch])
# plt.show()


pca3D = PCA(n_components=3, svd_solver='full')

pcax3D = pca3D.fit_transform(x)
pcax3D_test = pca3D.transform(x_test)

fig = plt.figure()
ax = Axes3D(fig)
plt.title('data after PCA (to 3D)')
ax.scatter(pcax3D[y==0][:,0],pcax3D[y==0][:,1],pcax3D[y==0][:,2], c='#0000FF')
ax.scatter(pcax3D[y==1][:,0],pcax3D[y==1][:,1],pcax3D[y==1][:,2], c='g')
ax.scatter(pcax3D[y==2][:,0],pcax3D[y==2][:,1],pcax3D[y==2][:,2], c='#FFFF00')
ax.scatter(pcax3D[y==3][:,0],pcax3D[y==3][:,1],pcax3D[y==3][:,2], c='#FFBF00')
ax.scatter(pcax3D[y==4][:,0],pcax3D[y==4][:,1],pcax3D[y==4][:,2], c='r')
ax.legend(handles=[blue_patch,green_patch,yellow_patch,orange_patch,red_patch])
plt.show()

# fig = plt.figure()
# ax = Axes3D(fig)
# plt.title('data after PCA (to 3D)')
# ax.scatter(pcax3D_test[y_test==0][:,0],pcax3D_test[y_test==0][:,1],pcax3D_test[y_test==0][:,2], c='#0000FF')
# ax.scatter(pcax3D_test[y_test==1][:,0],pcax3D_test[y_test==1][:,1],pcax3D_test[y_test==1][:,2], c='g')
# ax.scatter(pcax3D_test[y_test==2][:,0],pcax3D_test[y_test==2][:,1],pcax3D_test[y_test==2][:,2], c='#FFFF00')
# ax.scatter(pcax3D_test[y_test==3][:,0],pcax3D_test[y_test==3][:,1],pcax3D_test[y_test==3][:,2], c='#FFBF00')
# ax.scatter(pcax3D_test[y_test==4][:,0],pcax3D_test[y_test==4][:,1],pcax3D_test[y_test==4][:,2], c='r')
# ax.legend(handles=[blue_patch,green_patch,yellow_patch,orange_patch,red_patch])
# plt.show()

fig = plt.figure()
ax = Axes3D(fig)
plt.title('data after PCA (to 3D) 0-2 VS 3-4')
# ax.scatter(pcax3D_test[y_test<3][:,0],pcax3D_test[y_test<3][:,1],pcax3D_test[y_test<3][:,2], c='g')
# ax.scatter(pcax3D_test[y_test>2][:,0],pcax3D_test[y_test>2][:,1],pcax3D_test[y_test>2][:,2], c='r')
ax.scatter(pcax3D[y<3][:,0],pcax3D[y<3][:,1],pcax3D[y<3][:,2], c='g')
ax.scatter(pcax3D[y>2][:,0],pcax3D[y>2][:,1],pcax3D[y>2][:,2], c='r')
green_patch = mpatches.Patch(color='g', label='stage0-2')
red_patch = mpatches.Patch(color='red', label='stage3-4')
ax.legend(handles=[green_patch,red_patch])
plt.show()

fig = plt.figure()
ax = Axes3D(fig)
plt.title('data after PCA (to 3D) TESTvsTRAIN')
ax.scatter(pcax3D[:,0],pcax3D[:,1],pcax3D[:,2], c='g')
ax.scatter(pcax3D_test[:,0],pcax3D_test[:,1],pcax3D_test[:,2], c='r')
green_patch = mpatches.Patch(color='g', label='TRAIN')
red_patch = mpatches.Patch(color='r', label='TEST')
ax.legend(handles=[green_patch,red_patch])
plt.show()

# fig = plt.figure()
# ax = Axes3D(fig)
# plt.title('data after PCA (to 3D) AREA select')
# ax.scatter(pcax3D_test[A<M][:,0],pcax3D_test[A<M][:,1],pcax3D_test[A<M][:,2], c='g')
# ax.scatter(pcax3D_test[A>=M][:,0],pcax3D_test[A>=M][:,1],pcax3D_test[A>=M][:,2], c='r')
# green_patch = mpatches.Patch(color='g', label='small liver area')
# red_patch = mpatches.Patch(color='red', label='big liver area')
# ax.legend(handles=[green_patch,red_patch])
# plt.show()

# patient = ['MLS39','EK46','MM58','SHP43','AE50']
# blue_patch = mpatches.Patch(color='#0000FF', label=patient[0])
# green_patch = mpatches.Patch(color='g', label=patient[1])
# yellow_patch = mpatches.Patch(color='#FFFF00', label=patient[2])
# orange_patch = mpatches.Patch(color='#FFBF00', label=patient[3])
# red_patch = mpatches.Patch(color='red', label=patient[4])
# fig = plt.figure()
# ax = Axes3D(fig)
# plt.title('data after PCA (to 3D) patient from stage4')
# ax.scatter(pcax3D[PID==patient[0]][:,0],pcax3D[PID==patient[0]][:,1],pcax3D[PID==patient[0]][:,2], c='#0000FF')
# ax.scatter(pcax3D[PID==patient[1]][:,0],pcax3D[PID==patient[1]][:,1],pcax3D[PID==patient[1]][:,2], c='g')
# ax.scatter(pcax3D[PID==patient[2]][:,0],pcax3D[PID==patient[2]][:,1],pcax3D[PID==patient[2]][:,2], c='#FFFF00')
# ax.scatter(pcax3D[PID==patient[3]][:,0],pcax3D[PID==patient[3]][:,1],pcax3D[PID==patient[3]][:,2], c='#FFBF00')
# ax.scatter(pcax3D[PID==patient[4]][:,0],pcax3D[PID==patient[4]][:,1],pcax3D[PID==patient[4]][:,2], c='red')
# ax.legend(handles=[blue_patch,green_patch,yellow_patch,orange_patch,red_patch])
# plt.show()