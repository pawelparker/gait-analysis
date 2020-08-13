#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn import preprocessing


# In[2]:


df=pd.read_csv('gait.csv')


# In[3]:


df


# In[4]:


null_columns=df.columns[df.isnull().any()]
df[null_columns].isnull().sum()


# In[5]:


print(df[df.isnull().any(axis=1)][null_columns].head())


# In[6]:


df.dropna(axis=0, how='any', inplace=True)
df[null_columns].isnull().sum()


# In[7]:


df.info()


# In[8]:


df['activity'].value_counts().plot(kind = 'bar', title = "Activities")
plt.plot()


# In[9]:


df['uid'].value_counts().plot(kind='bar',title='Training Examples w.r.t Subjects(uid)')
plt.show()


# In[10]:


def plot_activity(activity, data):

    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3,
         figsize=(15, 10),
         sharex=True)
    plot_axis(ax0, data['time'], data['ax'], 'X-Axis')
    plot_axis(ax1, data['time'], data['ay'], 'Y-Axis')
    plot_axis(ax2, data['time'], data['az'], 'Z-Axis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()

def plot_axis(ax, x, y, title):

    ax.plot(x, y, 'r')
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)

for activity in np.unique(df['activity']):
    subset = df[df['activity'] == activity][:180]
    plot_activity(activity, subset)


# In[11]:


df


# In[12]:


df=df.drop(['uid', 'time','temp'], axis = 1)


# In[13]:


df


# In[14]:


from sklearn.utils import shuffle
df = shuffle(df)

#Train-Test Spillting
from sklearn.model_selection import train_test_split

train_set,test_set=train_test_split(df,test_size=0.15,random_state=42)


# In[15]:


print(f"Rows in train set:{len(train_set)}\nRows in test set:{len(test_set)}\n")


# In[16]:


train_set_labels=train_set['activity'].copy()
train_set=train_set.drop('activity',axis=1).copy()
test_set_labels=test_set['activity'].copy()
test_set=test_set.drop('activity',axis=1).copy()


# In[17]:


#Normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))
train_set = scaler.fit_transform(train_set)
scaler = MinMaxScaler(feature_range=(-1, 1))
test_set = scaler.fit_transform(test_set)


# In[18]:


#Spiltting into Train, Validation and Test
train_set,val_set,train_set_labels,val_set_labels=train_test_split(train_set,train_set_labels,test_size=0.10,random_state=42)


# In[19]:


print(f"Rows in train set:{len(train_set)}\nRows in validation set:{len(test_set)}\nRows in test set:{len(val_set)}\n")


# In[20]:


#Encoding the target variable
from sklearn.preprocessing import LabelEncoder
train_set_labels = LabelEncoder().fit_transform(train_set_labels)
val_set_labels = LabelEncoder().fit_transform(val_set_labels)
test_set_labels = LabelEncoder().fit_transform(test_set_labels)


# In[21]:


from elm import ELM
elm = ELM(hid_num=5000).fit(train_set, train_set_labels)
print("ELM Training Accuracy %0.3f " % elm.score(train_set,train_set_labels))
print("ELM Validation Accuracy %0.3f " % elm.score(val_set,val_set_labels))
print("ELM Testing Accuracy %0.3f " % elm.score(test_set,test_set_labels))
validation_prediction=elm.predict(val_set)
print(classification_report(val_set_labels, validation_prediction))
sns.heatmap(confusion_matrix(val_set_labels, validation_prediction),annot=True,fmt="d")


# In[22]:


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


# In[23]:


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


# In[24]:


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


# In[27]:


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


# In[34]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams.update({'figure.figsize':(8,5), 'figure.dpi':80})
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
plt.bar(x,y)
plt.xlabel('Testing Accuracy')
plt.ylabel("Values")
plt.title('2)TESTING')
plt.show()


# In[30]:


# Figure Size 
fig, ax = plt.subplots(figsize =(16, 9)) 
a=svc_score_val
b=round(elm.score(val_set,val_set_labels)*100,2)
c=random_forest_score_val
d=mlp_score_val
y=[a,b,c,d]
x=['SVC','ELM','RandomForest','MultilayerPerceptron'] 
ax.barh(x,y)  
for s in ['top', 'bottom', 'left', 'right']: 
    ax.spines[s].set_visible(False) 
  
ax.xaxis.set_ticks_position('none') 
ax.yaxis.set_ticks_position('none')   
ax.xaxis.set_tick_params(pad = 5) 
ax.yaxis.set_tick_params(pad = 10)  
ax.grid(b = True, color ='grey', 
        linestyle ='-.', linewidth = 0.5, 
        alpha = 0.2)  
ax.invert_yaxis() 
for i in ax.patches: 
    plt.text(i.get_width()+0.2, i.get_y()+0.5,  
             str(round((i.get_width()), 2)), 
             fontsize = 10, fontweight ='bold', 
             color ='grey') 
ax.set_title('VALIDATION ACCURACY', 
             loc ='center',fontsize = 20, fontweight ='bold' ) 
fig.text(0.9, 0.15, '@PawelParker', fontsize = 12, 
         color ='grey', ha ='right', va ='bottom', 
         alpha = 0.7) 
plt.show() 


# In[ ]:





# In[ ]:




