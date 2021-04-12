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


from sklearn.ensemble import RandomForestClassifier
'''
# FINDING BEST PARAMETERS FOR RANDOM FOREST CLASSIFIER
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)

rf_random.best_params_
'''

# USING THE BEST PARAMETERS FOR RANDOM FOREST CLASSIFIER
classifier = RandomForestClassifier(n_estimators = 5000, criterion = 'entropy', min_samples_split = 5, min_samples_leaf = 4, max_features = 'sqrt', max_depth = 80,bootstrap = True)
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