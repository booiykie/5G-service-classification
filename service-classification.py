#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  7 09:37:17 2022

@author: wisani
"""

#%% import required libraries

import pandas as pd
import numpy as np
import timeit
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import metrics, datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import seaborn as sns
from scipy import stats

import random
from sklearn import model_selection
from sklearn import metrics
#%% load dataset
# load the manufactured dataset/database
# dataset file is in the same directory

df = pd.read_csv('database.csv')
print("The size of the database is:", df.shape)

#df['Latency (ms)'] = np.log(df['Latency (ms)'])
plt.figure()
sns.pairplot(df)

#%% plot service groups


traffic_by_service = df.groupby("Service")["Latency (ms)"].count()

traffic_by_service.plot(kind = "bar", stacked = True)

#%% Exploring the dataset and infered schema

print("Dataset schema:\n", df.dtypes)  # all contables expected to be floats

# peek the data
pd.set_option('display.max_seq_items', None)
print(df.head(10))
print(df.columns)

print(df.loc[[10]])
print(df.describe())

#%%
scaler = StandardScaler()
X = df.drop('Service', axis=1)
Y = df['Service']

X = scaler.fit_transform(X)

dfx = pd.DataFrame(data=X, columns=df.columns[:-1])
print(dfx.describe())

from sklearn.decomposition import PCA
pca = PCA(n_components=None)
dfx_pca = pca.fit(dfx)

dfx_trans = pca.transform(dfx)

plt.figure()
plt.scatter(x=[i+1 for i in range(len(dfx_pca.explained_variance_ratio_))],
            y=dfx_pca.explained_variance_ratio_,
            s=200, alpha=0.75, c='red', edgecolor='m')
plt.grid(True)
plt.title('Explained variance ratio of \nfitted pc vector\n', fontsize=22)
plt.xlabel('Principal component', fontsize=15)
plt.xticks([i+1 for i in range(len(dfx_pca.explained_variance_ratio_))], fontsize=13)
plt.yticks(fontsize=15)
plt.ylabel("Variance ratio", fontsize=15)
plt.show()

    
#%% stack plot

for columnName in df.columns[:-1]:
    plt.figure()
    serviceData = df.groupby([columnName, "Service"])[columnName].count().unstack('Service').fillna(0)
    serviceData[[
                " UHD_Video_Streaming", 
                " Immerse_Experience", 
                " Vo5G", 
                " e_Health", 
                " ITS", 
                " Surveillance", 
                " Connected_Vehicles", 
                " Smart_Grid",
            ]].plot(kind='bar', stacked=True)
    print(serviceData)


#%% Data Preps
## Defining variables

X = df.iloc[:,:8].values  # KPI parameters
Y = df.iloc[:,-1].values  # Services labels

# Dataset splitting into trainig and test 80/20 split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 42)
print("The test data is:", X_test.shape)
print("The train data is:", X_train.shape)


#%% defining the supervised learning algorithms
# Creating a list of classifiers to infuse the respective algorithms

classifiers = [
    DecisionTreeClassifier(criterion = 'entropy', random_state = 0),                     
    RandomForestClassifier(n_estimators = 5, criterion = 'entropy', random_state = 0),    
    SVC(kernel = "linear", C = 0.025)
]  

names = ["Decision Tree", "Random Forest", "Linear SVM"]

#%% validation

# For the validation of the algorithm we use cross-validation technique with 10 division and accuracy it´s checked
kf = KFold(n_splits = 20)
cross_val_scores = [] # We create an empty tuple that will contain the cross-validation values means of each model
for name, model in zip(names, classifiers):
    starttime = timeit.default_timer()
    model.fit(X_train, y_train)            # We fit each model
    score = model.score(X_train, y_train)  # It is checked if the model learned well, testing its accuracy on the same training data
    print("{} training time : {}".format(name, timeit.default_timer() - starttime))
    print("The", name,"model metric accuracy is:", score*100, "%")
    cros_val_scores = cross_val_score(model, X_train, y_train, cv = kf, scoring = "accuracy") # We use K-Fold cross-validation
    print("Metrics of cross-validation of the", name, cros_val_scores)
    print("The Cross-validation mean of the", name,"is:", cros_val_scores.mean()*100, "%")
    cross_val_scores.append(cros_val_scores.mean())    # Values of cross-validation stage are going to be inserted in the tuple

    print("------------------------------------------------------------------------------------------------") 


#%% testing 

# Model Testing block.

y_predic_models = []   # We create an empty tuple that will contain the values of the predictions made by each model.
accuracy_models = []   # We create an empty tuple that will contain the values of the accuracy made by each model.
mcc_models = []        # We create an empty tuple that will contain the values of the Matthews coefcient of each model.

for name, model in zip(names, classifiers):
    y_predic = model.predict(X_test)              # We predict the test data (Xtest)
    y_predic_models.append(model.predict(X_test))  # The predictions are going to be inserted in the tuple 
    matrix = confusion_matrix(y_test, y_predic)  # Confusion Matrix of each model
    print("This is the confusion matrix result of the 1st Simulation with a", name,"\n", matrix, "\n")
    # Accuracy and Matthews coeficient of each model
    accuracy = accuracy_score(y_test, y_predic)   # Model accuracy of each model 
    accuracy_models.append(accuracy)   # The values of accuracy are going to be inserted in tuple
    print("The", name, "model has an accuracy of:", accuracy*100, "%")
    mcc = matthews_corrcoef(y_test, y_predic)      # Matthews coeficient
    mcc_models.append(mcc)               # The values of Matthews coeficient are going to be inserted in tuple
    print("The", name, "model has a Matthews coeficient of:", mcc*100, "%")
    print("--------------------------------------------------------------------------\n")
    
#%% results

# All the metrics and their values

for name, y_predic in zip(names, y_predic_models):
    print("The metrics of the", name, "model is:\n")
    print(metrics.classification_report(y_test, y_predic, digits = 3))


#%% validation
def calculate_difference():   # A function to calculate if the predictive model is overfitting
    global difference
    global overfitting
    global accuracy
    accuracy_cross_validation = cross_val_scores[1]*100  # We calculate difference for the Random Forest algorithm
    accuracy = accuracy_models[1]*100
    difference = accuracy_cross_validation - accuracy   # Difference between the accuracy of the cross-validation stage and the accuracy of the predictive model
    
    if difference > 5.5 or accuracy == 100:   # If difference > 5 or accuracy = 100% the model is overfitting
        overfitting = True
        print("It's look like the predictive model is overfitting\n")
    elif accuracy < accuracy_models[1]*100:   # We compared with the accuracy of the Decision Tree in the first simulation
        print("It's look like the predictive model is not good\n")
        overfitting = True
    else:    # else the model is not overfitting and capable for classify services
        overfitting = False
        
calculate_difference()

#%% plotting


# Feature names of the first simulation
feature_names = ['Latency (ms)', 'Jitter (ms)', 'Bit Rate (Mbps)', 'Packet Loss Rate (%)', 
                 'Peak Data Rate DL (Gbps)', 'Peak Data Rate UL (Gbps)', 'Mobility (km/h)',
                 'Reliability (%)']

# Class names 
class_names = ['UHD video streaming', 'Immerse experience', 'Connected vehicles', 'eHealth',
              'Industry automation', 'Smart grid', 'Video surveillance', 'ITS', 'Vo5G']

plt.figure(figsize=(22,10))        # Figure size
estimator = classifiers[1]        # Remember that the second value of the tuple corresponds to the Random Forest algorithm
estimator.fit(X_train, y_train)   # We train it again with the KPIs training data.
# The forest is trained in the same way since we are indicating the parameter of random_state = 0

# Now we draw the scheme
tree.plot_tree(estimator[2], feature_names = feature_names, fontsize = 9,   
               filled = True, rounded = True, class_names = class_names)

plt.savefig('tree', dpi = 100)   # We save the 1st scheme in .PNG format






