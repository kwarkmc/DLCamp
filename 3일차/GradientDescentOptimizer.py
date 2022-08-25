#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf

# In[2]:


tf.set_random_seed(777)


# In[4]:


x_train = [1, 2, 3]
y_train = [1, 2, 3]


# In[5]:


w = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')


# In[6]:


hypothesis = x_train * w + b


# In[7]:


cost = tf.reduce_mean(tf.square(hypothesis))


# In[9]:


train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)


# In[10]:


W_val = []
Cost_val = []


# In[13]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(20):
        val, cost_val, w_val, b_val = sess.run([train, cost, w, b])
        W_val.append(w_val)
        Cost_val.append(cost_val)
        print(step, cost_val, w_val, b_val)


# In[14]:


import matplotlib.pyplot as plt


# In[15]:


plt.plot(W_val, Cost_val, 'ro')
plt.ylabel('Cost')
plt.xlabel('W')
plt.show()


# In[ ]:




