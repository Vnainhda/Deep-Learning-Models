#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
import seaborn as sns


# In[2]:


import warnings 
warnings.filterwarnings('ignore')

df = pd.read_csv("./sonar_dataset.csv", header = None)
df.sample(4)


# In[3]:


#Data Exploration


# In[4]:


df.shape


# In[5]:


df.isna().sum()


# In[6]:


df.columns


# In[11]:


df[60].value_counts()


# In[7]:


#X and Y


# In[9]:


X = df.drop(60, axis= 'columns')
y = df[60]
y


# In[14]:


y = pd.get_dummies(y, drop_first=True)
y


# In[15]:


y.value_counts()


# In[16]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state =1)


# In[17]:


X_train.shape,X_test.shape


# In[18]:


import tensorflow 
from tensorflow import keras


# In[22]:


model = keras.Sequential([
    keras.layers.Dense(60,input_dim=60, activation = 'relu'),
    keras.layers.Dense(30, activation = 'relu'),
    keras.layers.Dense(15, activation = 'relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100, batch_size = 8)


# In[23]:


model.evaluate(X_test , y_test)


# In[24]:


y_pred = model.predict(X_test).reshape(-1)
print(y_pred[:10])

y_pred = np.round(y_pred)
print(y_pred[:10])


# In[25]:


y_test[:10]


# import sklearn

# In[29]:


import sklearn.metrics


# In[30]:


from sklearn.metrics import confusion_matrix,classification_report
print(classification_report(y_test,y_pred))


# In[35]:


modeld = keras.Sequential([
    keras.layers.Dense(60,input_dim=60, activation = 'relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(30, activation = 'relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(15, activation = 'relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100, batch_size = 8)


# In[36]:


model.evaluate(X_test,y_test)


# In[39]:


y_pred = model.predict(X_test).reshape(-1)
y_pred = np.round(y_pred)
from sklearn.metrics import confusion_matrix,classification_report
print(classification_report(y_test,y_pred))


# In[ ]:




