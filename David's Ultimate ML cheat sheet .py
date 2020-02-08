#!/usr/bin/env python
# coding: utf-8

# #pre-processing 
# #Data preprocessing is an important step in the data mining process. 
# #This is particularly applicable to data mining and machine learning projects.
# #Data-gathering methods are often loosely controlled, resulting in out-of-range values, impossible data combinations, missing values, etc
# #about 70% of your time building an ML algorithm would be spent doing data preprocessing.

# In[2]:


#importing libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
# you would still have to import more libraries when building an ML model e.g SKlearn,model_selection,XGboost,NLP.
#the models you would import varies with the task. however, the modle importted baove is sufficient to preprocess the data.


# In[ ]:


#importing the dataset 
dataset = pd.read_csv('Data.csv') #pd.read_csv is from the pandas library
#investigate which column has more missing value
nulls = dataset.isnull().sum()[dataset.isnull().sum() > 0].sort_values(ascending=False).to_frame().rename(columns={0: "MissingVals"})
nulls["MissingValsPct"] = nulls["MissingVals"] / len(train)
nulls
#splitting into x and y
x = dataset.iloc[:,].values# here you assigning the independent valriable/s to x 
y = dataset.iloc[:,].values# here you assigning the dependent valriable/s to y
#before the comma means all the rows and after the comma you select with column you want to work with. 
#you might have to change the import fuction depending on the data type you are importing.
# you can always add more conditionsbefore importing e.g delimiter ='\t' for tsv files(this type of file import is used during NLP)
# you can always check more condition by pressing shift + tab in jupyter note book or CMD/CTRL + i on spyder
#N/B indexing in python starts from 0 


# In[3]:


#Dealing with missing data
#1. you can drop it
#2. take the mean of columns 
#3. take the median of columns
#4. take the most frequent value 
x = x.dropna()#to drop missing data 

from sklearn.preprocessing import Imputer 
imputer = Imputer(missing_values='NaN',stragery = 'mean', axis = 0)#to use the mean of the column
imputer = imputer.fit(x[:,])#fitting th eimputer to the data
x[:,] = imputer.transform(x[:,])#transforming the data
#N/B:before the comma means all the rows and after the comma you select with column you want to work with. 

imputer = Imputer(missing_values='NaN',stragery = 'median', axis = 0)#to take the median of the column
imputer = imputer.fit(x[:,])#fitting th eimputer to the data
x[:,] = imputer.transform(x[:,])#transforming the data 
#N/B:before the comma means all the rows and after the comma you select with column you want to work with. 

imputer = Imputer(missing_values='NaN',stragery = 'most_frequent', axis = 0)#to take the most frequent/mode of the column
imputer = imputer.fit(x[:,]) #fitting th eimputer to the data
x[:,] = imputer.transform(x[:,])#transforming the data 
#N/B:before the comma means all the rows and after the comma you select with column you want to work with. 


# In[ ]:


#Encoding categorical data 
#1.labelEncoder 
#2.OneHotEncoder

#label encoder
g= x.iloc[:,]# in this case g is the column that we want to encode 
from sklearn import preprocessing 
le = preprocessing.LabelEncoder()
le.fit(g)
g=le.transform(g)
x['column _name']=g
#however, the machine might think a value is more important than the other which is not true in most cases so  using a onehotencoder might be better.

#one hot encoder 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder() # created an object for the function 
x[:,] = labelencoder_X.fit_transform(x[:,])# select the column you want to encode
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()

#dummy varibale trap



# In[ ]:


#splitting into test and train
from sklearn.model_selection import train_test_split #importing the library 
x_train,x_test,y_train,y_test = test_train_split(x, y, test_size=0.2, random_state=0)#splitting: test:20% train:80%


# #feature Scaling 
# Feature scaling is a method used to normalize the range of independent variables or features of data. 
# In data processing, it is also known as data normalization and is generally performed during the data preprocessing step.
# 
# 

# In[ ]:


from sklearn.preprocessing import StandardScaler#importing the library 
sc_X = StandardScaler()#creating an object
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)
sc_y = StandardScaler()#creating an object
y_train = sc_y.fit_transform(y_train)
#its not used everytime but its still something to have in your ML arsenal.


# # Regression 
# 1.Linear regression 
# 2.Multilinear regression 
# 3.polynomial regression 
# 4.support vector regression 
# 5.decision tree regression 
# 6. random forest regression 

# In[ ]:


#Linear Regression 
from sklearn.linear_model import LinearRegression
MLR = LinearRegression()
MLR.fit(x_train,y_train)
y_pred = MLR.predict(x_test)


# Backward elimination (or backward deletion) is the reverse process. 
# All the independent variables are entered into the equation first and each one is deleted one at a time if they do not contribute to the regression equation. 
# Stepwise selection is considered a variation of the previous two methods
# step 1 : Select a significant level 
# 2: fit the model with all the predictors
# 3:consider the predictor with the highest value. if the value is greater tahn the initial significant level, remove it if not your have gotten the optimal amount of independent variable.

# In[ ]:


#MultiLinear Regression 
from sklearn.linear_model import LinearRegression
MLR = LinearRegression()
MLR.fit(x_train,y_train)
y_pred = MLR.predict(x_test)
# the difference between a multilinear regression and the regular linear regression is beacuse we would be using a backward elimination method.
#the bacward elimination method is used to remove columns of less importnace to increase the accuracy of the model.
# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
x = np.append(arr = np.ones((, )).astype(int), values = x, axis = 1)
x_opt = x[:, []]# the indexes of the columns should be in the []
MLR_OLS = sm.OLS(endog = y, exog = x_opt).fit()
MLR_OLS.summary()
x_opt = x[:, []]
MLR_OLS = sm.OLS(endog = y, exog = x_opt).fit()
MLR_OLS.summary()


# In[ ]:


#Polynomial Regression

