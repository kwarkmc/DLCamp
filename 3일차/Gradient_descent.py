#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf

# In[2]:


tf.set_random_seed(777)


# In[3]:


x_data = [1, 2, 3]
y_data = [1, 2, 3]


# In[4]:


w = tf.Variable(tf.random_normal(([1]), name = 'weight'))


# In[5]:


X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)


# In[6]:


hypothesis = X * w


# In[7]:


cost = tf.reduce_mean(tf.square(hypothesis - Y))
learning_rate = 0.1
gradient = tf.reduce_mean((w * X - Y) * X)
descent = w - learning_rate * gradient
update = w.assign(descent)


# In[11]:


W_val = []
Cost_val = []


# In[12]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(20):
        val, cost_val, w_val = sess.run([update, cost, w], feed_dict = {X : x_data, Y : y_data})
        W_val.append(w_val)
        Cost_val.append(cost_val)
        print(step, cost_val, w_val)


# In[14]:


import matplotlib.pyplot as plt

plt.plot(W_val, Cost_val, 'ro')
plt.ylabel('Cost')
plt.xlabel('W')
plt.show()


# In[ ]:




