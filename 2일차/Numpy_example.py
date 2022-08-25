#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

# In[2]:


x = np.array([1, 2, 3])


# In[3]:


x


# In[4]:


type(x)


# In[5]:


x.shape


# In[6]:


y = range(10)


# In[7]:


y


# In[8]:


y = np.arrange(10)


# In[9]:


y = np.arange(10)


# In[10]:


y


# In[11]:


x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])


# In[12]:


x + y


# In[13]:


x * y


# In[14]:


x / 2.0


# In[15]:


A = np.array([[1, 2], [3, 4]])


# In[16]:


print(A)


# In[17]:


A.shape


# In[18]:


A.dtype


# In[20]:


B = np.array([[3.0, 0.6]])


# In[21]:


print(A+B)


# In[22]:


print(A*B)


# In[ ]:




