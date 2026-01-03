#!/usr/bin/python3
# filename: fileIO.py


import numpy as np

def dataread2d( filePath, dtype, shape ):
    with open(filePath, 'rb') as f:
        matrix = np.zeros(shape, dtype=dtype)
        matrix = np.fromfile(f, dtype=dtype)
        matrix = matrix.reshape(shape[1], shape[0])
        matrix = np.transpose(matrix)
        return matrix

def datawrite2d( filePath, matrix):
    matrix = np.transpose(matrix)
    byteData=matrix.tobytes()
    with open(filePath, 'wb') as f:
        f.write(byteData)




