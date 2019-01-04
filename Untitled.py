#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


labeled_images = pd.read_csv('./train.csv')


# In[6]:


labeled_images.shape


# In[7]:


images = labeled_images.iloc[0:5000,1:]


# In[8]:


images.shape


# In[17]:


labeled_images.iloc[:6,:6]


# In[18]:


labels = labeled_images.iloc[0:5000,:1]


# In[22]:


labels.head(30)


# In[26]:


train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)


# In[27]:


train_images.shape


# In[28]:


test_images.shape


# In[29]:


train_labels.shape


# In[31]:


i=1
img=train_images.iloc[i].as_matrix()


# In[32]:


img


# In[33]:


i=1
img=train_images.iloc[i]


# In[35]:


img.shape


# In[37]:


i=1
img=train_images.iloc[i].as_matrix()


# In[ ]:




