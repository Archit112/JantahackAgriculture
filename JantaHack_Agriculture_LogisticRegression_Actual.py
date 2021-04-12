# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 21:29:41 2020

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
#Y.unique() 


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


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# PREDICTING THE RESULT
y_pred = classifier.predict(X_test)
y_pred = y_pred.astype(int)
#y_test = labelencoder_X.fit_transform(y_test)

y_pred = np.array(y_pred)



'''
# CHECKING THE ACCURACY OF THE PREDICTION
# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Finding the acuracy score
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test ,y_pred)*100
'''

# GETTING THE PREDICITON FILE READY FOR EXPORT TO CSV
export = pd.DataFrame(testset, columns = ['ID']) 
export['Crop_Damage'] = pd.DataFrame({'Crop_Damage': y_pred})
#export['Crop_Damage'].value_counts()
export.to_csv(r'Test_File_new.csv', index = False)