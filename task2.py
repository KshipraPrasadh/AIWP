# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 10:24:28 2019

@author: BATCH1
"""

import numpy as np
import matplotlib.pyplot as plt
#from sklearn import cross_validation
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from utilities import visualize_classifier
from sklearn import tree
import graphviz

input_file='data_prblm1.txt'
data= np.loadtxt(input_file,delimiter=',')
X,y= data[:,:-1],data[:,-1]

#seperate input data into two classes based on labels
class_0= np.array(X[y==0])
class_1= np.array(X[y==1])

params={'random_state':0,'max_depth':4}
classifier=DecisionTreeClassifier(**params)
classifier.fit(X,y)

from sklearn.tree import export_graphviz
export_graphviz(classifier,out_file='tree1.dot',class_names=['Strep throat','Allergy','Cold'],rounded=True,proportion=False,precision=2,filled=True)

