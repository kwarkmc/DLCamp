#!/usr/bin/env python
# coding: utf-8

# In[8]:


import tensorflow as tf
import numpy as np

# In[9]:


tf.set_random_seed(777)


# In[10]:


xy = np.loadtxt('data-03-diabetes.csv', delimiter = ',', dtype=float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]


# In[11]:


print(x_data.shape, y_data.shape)


# In[12]:


X = tf.placeholder(tf.float32, shape = [None, 8])
Y = tf.placeholder(tf.float32, shape = [None, 1])


# In[13]:


W = tf.Variable(tf.random_normal([8, 1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')


# In[14]:


hypothesis = tf.sigmoid(tf.matmul(X, W) + b)


# In[15]:


cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))


# In[16]:


train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)


# In[17]:


predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype = float32))


# In[19]:


Cost_val = []
Steps = []


# In[22]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(10001):
        cost_val, val = sess.run([cost, train], feed_dict = {X : x_data, Y : y_data})
        Cost_val.append(cost_val)
        Steps.append(step)
        if step % 200 == 0:
            print(step, cost_val)
            
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict = {X : x_data, Y : y_data})
    print("\nHypothesis : ", h, "\nCorrect (Y) : ", c, "\nAccuracy : ", a)


# In[23]:


import matplotlib.pyplot as plt


# In[25]:


plt.plot(Steps, Cost_val)
plt.xlabel('Step')
plt.ylabel('Cost')
plt.show


# In[ ]:




