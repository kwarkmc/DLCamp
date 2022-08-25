#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt

# In[2]:


X = [1., 2., 3.]
Y = [1., 2., 3.]
m = n_samples = len(X)


# In[3]:


W = tf.placeholder(tf.float32)


# In[4]:


hypothesis = tf.multiply(X, W)


# In[5]:


cost = tf.reduce_sum(tf.pow(hypothesis - Y, 2)) / (m)


# In[6]:


init = tf.global_variables_initializer()


# In[7]:


W_val = []
cost_val = []


# In[8]:


sess = tf.Session()
sess.run(init)


# In[9]:


for i in range(-30, 50):
    print(i * 0.1, sess.run(cost, feed_dict = {W : i * 0.1}))
    W_val.append(i * 0.1)
    cost_val.append(sess.run(cost, feed_dict = {W : i * 0.1}))


# In[10]:


plt.plot(W_val, cost_val, 'ro')
plt.ylabel('Cost')
plt.xlabel('W')
plt.show()


# In[ ]:




