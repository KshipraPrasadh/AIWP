# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 10:58:53 2019

@author: batch1
"""

import numpy as np
import matplotlib.pyplot as plt
generate_random=np.random.RandomState(0)
x=10*generate_random.rand(100)
print(x.shape)
y=3*x+np.random.randn(100)
y.shape
plt.figure(figsize=(10,8))
plt.scatter(x,y);
from sklearn.linear_model import LinearRegression
model=LinearRegression(fit_intercept=True)
model
X=x.reshape(-1,1)
print("x",x.shape)
print("X",X.shape)
model.fit(X,y)
print(model.coef_)
print(model.intercept_)
x_fit=np.linspace(-1,11)
X_fit=x_fit.reshape(-1,1)
y_fit=model.predict(X_fit)
plt.figure(figsize=(10,8))
plt.scatter(x,y);
plt.plot(x_fit,y_fit);