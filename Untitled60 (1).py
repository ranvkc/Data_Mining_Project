#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import gc


# In[10]:


# read the dataset
digits = pd.read_csv("/Users/Admin/Desktop/Likith/Data Mining/train.csv")
digits.head(20)


# In[11]:


digits.columns


# In[12]:


digits.info()


# In[13]:


digits.describe()


# In[14]:


one = digits.iloc[2,1:]
one.shape


# In[15]:


one = one.values.reshape(28,28)
plt.imshow(one, cmap='gray')


# In[16]:


eight = digits.iloc[10,1:]
eight.shape


# In[17]:


eight = eight.values.reshape(28,28)
plt.imshow(eight, cmap='gray')


# In[18]:


print(one[5:-5, 5:-5])


# In[19]:


print(eight[5:-5, 5:-5])


# In[20]:


# summarise the counts of 'label' to see how many labels of each digit are present
digits.label.astype('category').value_counts()


# In[21]:


#Summarise count in terms of percentage. This tells us if it is balanced dataset or not
100*(round(digits.label.astype('category').value_counts()/len(digits.index),4))


# In[22]:


# Check if there are any missing values
digits.isnull().sum()


# In[23]:


# average values/distributions of features
description = digits.describe()
description


# In[24]:


#Splitting data into train and test 
X = digits.iloc[:, 1:]
y = digits.iloc[:, 0]
y


# In[25]:


# Rescaling the features
from sklearn.preprocessing import scale
X = scale(X)
X


# In[28]:


# We are taking only 100% of the train data. 

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[29]:


#Building a linear SVM model (i.e, a linear kernel)
from sklearn import svm
from sklearn import metrics

svm_linear = svm.SVC(kernel='linear')

svm_linear.fit(X_train, y_train)


# In[20]:


predictions = svm_linear.predict(X_test)
predictions[:10]


# In[21]:


# Evaluating accuracy
confusion = metrics.confusion_matrix(y_true=y_test, y_pred = predictions)
confusion


# In[22]:


#Calculating accuracy
metrics.accuracy_score(y_true=y_test, y_pred=predictions)


# In[23]:


# class-wise accuracy
class_wise = metrics.classification_report(y_true=y_test, y_pred=predictions)
print(class_wise)


# In[24]:


#gc.collect() to free up memory

gc.collect()


# In[30]:


#Non-linear SVM
#rbf kernel with other hyperparameters kept to default

svm_rbf =svm.SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)


# In[31]:


#predict
predictions = svm_rbf.predict(X_test)
predictions[:10]


# In[32]:


# Evaluating accuracy
confusion = metrics.confusion_matrix(y_true=y_test, y_pred = predictions)
confusion


# In[33]:


#accuracy
print(metrics.accuracy_score(y_true=y_test, y_pred=predictions))


# In[34]:


# class-wise accuracy
class_wise = metrics.classification_report(y_true=y_test, y_pred=predictions)
print(class_wise)


# In[35]:


d = {'ImageId': np.arange(1,predictions.shape[0]+1), 'Label': predictions}
pd.DataFrame(d)


# In[37]:


from sklearn.model_selection import cross_val_score
metrics.accuracy_score = cross_val_score(estimator=svm.SVC(), X=X_test, y=y_test, cv=5)
print(metrics.accuracy_score)


# In[ ]:




