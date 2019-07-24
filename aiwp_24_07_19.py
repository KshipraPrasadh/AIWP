# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 10:08:35 2019

@author: BATCH1
"""

from scipy import linalg

import numpy as np
numpy_mat=np.array([[3,2,2],[5,5,2],[7,8,69]])
print(linalg.det(numpy_mat))
EW,EV=linalg.eig(numpy_mat)
print(EW,EV)
from numpy import poly1d
p=poly1d([3,2,1])
print(p)
print(p*p)
print(p.deriv())

#fft
from scipy.fftpack import fft,ifft
x=np.array([1.0,2.0,3.0,4.0])
y=fft(x)
print(y)
print(ifft(y))

x=np.array([[1.0+1j,2.0+2j],[3.0+3j,4.0+4j]])
y=fft(x)
print(y)
print(ifft(y))

import pandas as pd
df=pd.DataFrame({'Marks':[44,55,66,46,34],'Name':['abc','def','ghi','jkl','mno']})
print(df.head())
print(df['Marks'].value_counts())
print(df['Name'].nunique())
print(df['Name'].unique())
print(type(df['Name'].unique()))
print(df.info())
print(df.describe())
df_2=df[df['Marks']>50]
print(df_2)
def pract_marks(data1):
    return data1+5
df_3=df['Marks'].apply(pract_marks)
print(df_3)

print(df.sort_values(by='Marks'))
print(df.isnull())

from numpy.random import randn

