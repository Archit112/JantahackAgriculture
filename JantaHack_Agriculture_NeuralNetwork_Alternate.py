# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 22:31:18 2020

@author: archi
"""

# ADDING RELAVENT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# DATA INITIALIZATION
dataset = pd.read_csv("train_yaOffsB.csv")
X = dataset.iloc[:, 1:9]
Y = dataset.iloc[:, 9]

#X["Number_Weeks_Used"].unique() 


# DATA PRE-PROCESSING FOR TRAINING FILES

# Handling misssing values
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values = np.nan , strategy = 'mean')
imp = imp.fit(X.iloc[:, 2:8])
X.iloc[:, 2:8] = imp.transform(X.iloc[:, 2:8])

    
plt.subplots(figsize=(20,15))    
sns.heatmap(dataset.corr(), annot=True)  
  
'''
X = X.drop(['Education'], axis=1)
X = X.drop(['Self_Employed'], axis=1)
X = X.drop(['ApplicantIncome'], axis=1)
X = X.drop(['CoapplicantIncome'], axis=1)
X = X.drop(['LoanAmount'], axis=1)
X = X.drop(['Loan_Amount_Term'], axis=1)

Y = dataset['Loan_Status']
X = X.iloc[:, 0:5]
'''

# ADDING TEST FILES
X_train = X
y_train = Y

testset = pd.read_csv("test_pFkWwen.csv")
X_test = testset.iloc[:, 1:9]
# DATA PRE-PROCESSING FOR TEST FILES

#X_test["Number_Weeks_Used"].unique() 

# Handling misssing values
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values = np.nan , strategy = 'mean')
imp = imp.fit(X_test.iloc[:, 2:8])
X_test.iloc[:, 2:8] = imp.transform(X_test.iloc[:, 2:8])

'''
X_test = X_test.drop(['Education'], axis=1)
X_test = X_test.drop(['Self_Employed'], axis=1)
X_test = X_test.drop(['ApplicantIncome'], axis=1)
X_test = X_test.drop(['CoapplicantIncome'], axis=1)
X_test = X_test.drop(['LoanAmount'], axis=1)
X_test = X_test.drop(['Loan_Amount_Term'], axis=1)
'''

# FEATURE SCALING
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_test = sc.fit_transform(X_test)
X_train = sc.fit_transform(X_train)

'''
# SPLITTING DATA INTO TEST AND TRAINING SET
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
'''

import keras
classifier = keras.Sequential([keras.layers.Dense(4, input_dim=8)])
classifier.compile(optimizer = 'sgd', loss = 'mean_squared_error')

#Compiling the neural network
classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])

#Fitting the data to the training dataset
classifier.fit(X_train,y_train, batch_size=10, epochs=100)

'''
eval_model=classifier.evaluate(X_train, y_train)
eval_model
'''

y_pred=classifier.predict(X_test)
y_pred =(y_pred>0.5)

y_pred = y_pred.astype(int)
#y_test = labelencoder_X.fit_transform(y_test)

y_pred = np.array(y_pred)

'''
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test ,y_pred)*100
'''

export = pd.DataFrame(testset, columns = ['ID']) 
export['Crop_Damage'] = pd.DataFrame({'Crop_Damage': y_pred[:, 0]})
#export['Crop_Damage'].value_counts()
export.to_csv(r'Test_File_new_2.csv', index = False)
