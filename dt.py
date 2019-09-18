# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 16:10:17 2019

@author: 16BIS0008
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
#from sklearn import cross_validation
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from utilities import visualize_classifier
from sklearn import tree
import graphviz

input_file='data_decision_trees.txt'
data= np.loadtxt(input_file,delimiter=',')
X,y= data[:,:-1],data[:,-1]

#seperate input data into two classes based on labels
class_0= np.array(X[y==0])
class_1= np.array(X[y==1])

plt.figure()
plt.scatter(class_0[:,0],class_0[:,1],s=None,facecolors='black',edgecolors='blue',linewidth=1, marker='x')
plt.scatter(class_1[:,0],class_1[:,1],s=None,facecolors='white',edgecolors='red',linewidth=1, marker='o')
plt.title('Input data')

X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=5)


params={'random_state':0,'max_depth':4}
classifier= DecisionTreeClassifier(**params)
classifier.fit(X_train,y_train)
visualize_classifier(classifier,X_train,y_train,'Training dataset')

y_test_pred= classifier.predict(X_test)
visualize_classifier(classifier,X_test,y_test,'Test dataset')

#evaluate classifier performance
class_names = ['Class-0', 'Class-1']
print("\n" + "#"*70)
print("\nClassifier performance on training dataset\n")
print(classification_report(y_train, classifier.predict(X_train), target_names=class_names))
print("#"*70 + "\n")
      
print("\n" + "#"*70)
print("\nClassifier performance on test dataset\n")
print(classification_report(y_test, y_test_pred, target_names=class_names))
print("#"*70 + "\n")
      
plt.show()

from sklearn.tree import export_graphviz
#export as dot file
export_graphviz(classifier, out_file='tree1.dot', class_names=['0', '1'], rounded = True, proportion = False, precision = 2, filled = True)




