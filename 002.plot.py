#!/usr/bin/python3

import sys
import math
sys.path.append("./include/")

import numpy as np
import matplotlib.pyplot as plt
from myModule import fileIO 
from myModule import matching 

nt=500
nx=1501

fileOrig="./movie/movie.dat1600.000"

shape=[nt,nx]
origAll2d=fileIO.dataread2d(fileOrig,np.float32,shape)
#demAll2d=np.zeros([nt,nx],np.float32)

plt.rcParams['font.family'] = 'Times New Roman'
plt.figure(figsize=(10, 6), dpi=100)
plt.imshow(origAll2d, aspect='auto',cmap='gray',\
        interpolation='bicubic',vmin=1000, vmax=4000)
plt.xlabel("Distance (m)", fontsize=15, fontweight='bold')
plt.ylabel("Depth (m)", fontsize=15, fontweight='bold')
plt.title('Sound', y=1.01, fontsize=15, fontweight='bold')
plt.xticks(np.arange(0, nx, nx/5), size=14, fontweight='bold')
plt.yticks(size=14, fontweight='bold')
cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=14,)
plt.show()
#fileIO.datawrite2d("./data/dem.dat",demAll2d)




