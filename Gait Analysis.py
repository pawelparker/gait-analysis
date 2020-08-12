#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# # Data Importing, Preprocessing and Preparation

# In[2]:


df=pd.read_csv('gait_data.csv')


# In[3]:


df.head()


# In[4]:


df = df.drop(df.columns[[0]], axis=1) 


# In[5]:


df


# In[6]:


df['% Gait Cycle']=df['% Gait Cycle'].map(lambda x: x.rstrip("%") )


# In[7]:


#Train-Test Spillting
from sklearn.model_selection import train_test_split

train_set,test_set=train_test_split(df,test_size=0.15,random_state=42)


# In[8]:


print(f"Rows in train set:{len(train_set)}\nRows in test set:{len(test_set)}\n")


# In[9]:


train_set_labels=train_set['Position'].copy()
train_set=train_set.drop('Position',axis=1).copy()
test_set_labels=test_set['Position'].copy()
test_set=test_set.drop('Position',axis=1).copy()


# In[10]:


#Normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))
train_set = scaler.fit_transform(train_set)
scaler = MinMaxScaler(feature_range=(-1, 1))
test_set = scaler.fit_transform(test_set)


# In[11]:


#Spiltting into Train, Validation and Test
train_set,val_set,train_set_labels,val_set_labels=train_test_split(train_set,train_set_labels,test_size=0.10,random_state=42)


# In[12]:


print(f"Rows in train set:{len(train_set)}\nRows in validation set:{len(test_set)}\nRows in test set:{len(val_set)}\n")


# In[13]:


#Encoding the target variable
from sklearn.preprocessing import LabelEncoder
train_set_labels = LabelEncoder().fit_transform(train_set_labels)
val_set_labels = LabelEncoder().fit_transform(val_set_labels)
test_set_labels = LabelEncoder().fit_transform(test_set_labels)


# # Extreme Learning Machine (ELM) implementation
# The ELM algorithm is similar to other neural networks with 3 key differences:
# 
# 1.The number of hidden units is usually larger than in other neural networks that are trained using backpropagation.
# 
# 2.The weights from input to hidden layer are randomly generated, usually using values from a continuous uniform distribution.
# 
# 3.The output neurons are linear rather than sigmoidal, this means we can use least square errors regression to solve the output weights.
# 
# Installation:
# pip install git+https://github.com/masaponto/python-elm

# In[14]:


from elm import ELM
elm = ELM(hid_num=100000).fit(train_set, train_set_labels)
print("ELM Training Accuracy %0.3f " % elm.score(train_set,train_set_labels))
print("ELM Validation Accuracy %0.3f " % elm.score(val_set,val_set_labels))
print("ELM Testing Accuracy %0.3f " % elm.score(test_set,test_set_labels))
validation_prediction=elm.predict(val_set)
print(classification_report(val_set_labels, validation_prediction))
sns.heatmap(confusion_matrix(val_set_labels, validation_prediction),annot=True,fmt="d")


# In[15]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})
a=elm.score(train_set,train_set_labels)*100
b=elm.score(val_set,val_set_labels)*100
c=elm.score(test_set,test_set_labels)*100
y=[a,b,c]
x=['Training','Validation','Testing']
plt.bar(x,y)
plt.xlabel('Accuracy')
plt.ylabel("Values")
plt.title('Accuracy Comparision of ELM')
plt.show()


# # Ensemble Learning (Random Forest Classifier)

# In[16]:


from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(train_set, train_set_labels)
#Predict Output
rf_predicted = random_forest.predict(val_set)
random_forest_score = round(random_forest.score(train_set, train_set_labels) * 100, 2)
random_forest_score_val =round(random_forest.score(val_set, val_set_labels) * 100, 2)
random_forest_score_test = round(random_forest.score(test_set, test_set_labels) * 100, 2)
print('Random Forest Training Score: \n', random_forest_score)
print('Random Forest Validation Score: \n', random_forest_score_val)
print('Random Forest Test Score: \n', random_forest_score_test)
print('Accuracy: \n', accuracy_score(val_set_labels,rf_predicted))
print(classification_report(val_set_labels,rf_predicted))
sns.heatmap(confusion_matrix(val_set_labels,rf_predicted),annot=True,fmt="d")


# # Multilayer Perceptron (ANN)

# In[17]:


#MLP classifier
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(50,50,50),activation='tanh', max_iter=500, alpha=1e-4,
                     solver='lbfgs', verbose=10,  random_state=21,tol=0.000000001)
clf.fit(train_set,train_set_labels)
#Predict Output 
y_pred = clf.predict(val_set)
mlp_score = round(clf.score(train_set, train_set_labels) * 100, 2)
mlp_score_test = round(clf.score(test_set, test_set_labels) * 100, 2)
mlp_score_val =  round(clf.score(val_set, val_set_labels) * 100, 2)
print('MLP classifier Training Score: \n', mlp_score)
print('MLP classifier Validation Score: \n', mlp_score_val)
print('MLP classifier Test Score: \n', mlp_score_test)
print('Accuracy: \n', accuracy_score(val_set_labels,y_pred))
print(classification_report(val_set_labels,y_pred))
sns.heatmap(confusion_matrix(val_set_labels,y_pred),annot=True,fmt="d")


# # SVM Classifier

# In[19]:


from sklearn.svm import SVC
svc = SVC()
# Train the model using the training sets and check score
svc.fit(train_set, train_set_labels)
#Predict Output
svc_predicted= svc.predict(val_set)
svc_score = round(svc.score(train_set, train_set_labels) * 100, 2)
svc_score_val = round(svc.score(val_set, val_set_labels) * 100, 2)
svc_score_test = round(svc.score(test_set, test_set_labels) * 100, 2)
print('SVM Training Score: \n', svc_score)
print('SVM Validation Score: \n', svc_score_val)
print('SVM Test Score: \n', svc_score_test)
print('Accuracy: \n', accuracy_score(val_set_labels,svc_predicted))
print('Classification Report: \n', classification_report(val_set_labels,svc_predicted))

sns.heatmap(confusion_matrix(val_set_labels,svc_predicted),annot=True,fmt="d")


# # Model Comparision

# In[20]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams.update({'figure.figsize':(8,5), 'figure.dpi':120})
a=svc_score_val
b=round(elm.score(val_set,val_set_labels)*100,2)
c=random_forest_score_val
d=mlp_score_val
y=[a,b,c,d]
x=['SVC','ELM','RandomForest','MultilayerPereceptron']
plt.bar(x,y)
plt.xlabel('Validation Accuracy')
plt.ylabel("Values")
plt.title('1)VALIDATION')
plt.show()
a=svc_score_test
b=round(elm.score(test_set,test_set_labels)*100,2)
c=random_forest_score_test
d=mlp_score_test
y=[a,b,c,d]
x=['SVC','ELM','RandomForest','MultilayerPereceptron']
plt.bar(x,y)
plt.xlabel('Testing Accuracy')
plt.ylabel("Values")
plt.title('2)TESTING')
plt.show()
a=svc_score
b=round(elm.score(train_set,train_set_labels)*100,2)
c=random_forest_score
d=mlp_score
y=[a,b,c,d]
x=['SVC','ELM','RandomForest','MultilayerPereceptron']
plt.bar(x,y)
plt.xlabel('Training Accuracy')
plt.ylabel("Values")
plt.title('3)TRAINING')
plt.show()


# In[ ]:




