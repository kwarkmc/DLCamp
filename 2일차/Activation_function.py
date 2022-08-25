#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import matplotlib.pyplot as plt

# In[2]:


x = np.array([0.0, 1.0, 2.0])


# In[3]:


print(x)


# In[4]:


y = x > 0


# In[5]:


print(y)


# In[6]:


y = y.astype(np.int)
#type을 bool 에서 int로 바꾼다.


# In[7]:


print(y)


# Step Function (계단함수)

# In[12]:


def step_function(x):
    return np.array(x > 0, dtype=np.int)


# In[14]:


x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)


# In[15]:


plt.plot(x, y)
plt.ylim(-0.1, 1.1) #y축 범위 설정
plt.show()


# Sigmoid Function (시그모이드함수)

# In[16]:


def sigmoid(x):
    return 1/(1 + np.exp(-x))


# In[17]:


y = sigmoid(x)


# In[19]:


plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()


# ReLu (Rectified Linear Function) 함수

# In[20]:


def relu(x):
    return np.maximum(0, x)


# In[21]:


y = relu(x)


# In[27]:


plt.plot(x, y)
#plt.ylim(-1, 6)
plt.show()


# In[ ]:




