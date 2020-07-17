"""
Created on Sun Apr 19 15:39:27 2020

@author: subhrohalder
"""
#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#csv to DataFrame conversion
dataset= pd.read_csv('Churn_Modelling.csv')

#Data Selection 
y=dataset.loc[:,'Exited'].values
X=dataset.iloc[:,:]
X=X.drop(columns=['Exited','RowNumber','CustomerId','Surname'])

#One Hot encoding
X=pd.get_dummies(X,drop_first=True)

"""One hot encoding is done above but there is one 
issue we can fall in dummmy variable trap 
so we have to remove one extra column so we have used
 drop_first=True.There comes co relation
between the data because of that extra column"""

#Split Train and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.25)

#Standardization(Feature Scaling)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(X_train)
X_train=sc.transform(X_train)
X_test=sc.transform(X_test)

#LogisticRegression
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,y_train)

#Support Vector Machine
from sklearn.svm import SVC
model=SVC()
model.fit(X_train,y_train)

#Kernel Support Vector Machine
from sklearn.svm import SVC
model=SVC(kernel='rbf')
model.fit(X_train,y_train)

#DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(X_train,y_train)

#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=400,max_depth=6) #n_estimators is number of decision tree max_depth is maximum division model can do
model.fit(X_train,y_train)

#Prediction
y_pred=model.predict(X_test)

#Testing accuracy
from sklearn.metrics import accuracy_score
accuracy_on_test=accuracy_score(y_test,y_pred)

#Training Accuracy
y_pred_2=model.predict(X_train)
accuracy_on_train=accuracy_score(y_train,y_pred_2)

"""Training Accuracy should not be 100% because it means the
 model is overfitting so we are comparing the 
accuracy of trainning and Testing so that difference should
 not be large so try to equate both by tunning 
the parameters"""