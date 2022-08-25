#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np

# In[2]:


tf.set_random_seed(777)


# In[3]:


xy = np.loadtxt('data-01-test-score.csv', delimiter = ',', dtype = np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]


# In[4]:


print(x_data, "\nx_data type : ", x_data.shape)
print(y_data, "\ny_data type : ", y_data.shape)


# In[5]:


X = tf.placeholder(tf.float32, shape = [None, 3])
Y = tf.placeholder(tf.float32, shape = [None, 1])


# In[6]:


W = tf.Variable(tf.random_normal([3, 1]), name = 'Weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')


# In[7]:


hypothesis = tf.matmul(X, W) + b


# In[8]:


cost = tf.reduce_mean(tf.square(hypothesis - Y))


# In[9]:


optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)


# In[10]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())


# In[11]:


Cost_val = []
Steps = []


# In[12]:


for step in range(2001):
    cost_val, hy_val, val = sess.run([cost, hypothesis, train], feed_dict = {X : x_data, Y : y_data})
    Steps.append(step)
    Cost_val.append(cost_val)
    if step % 10 == 0:
        print(step, "Cost : ", cost_val, "\nPrediction : \n", hy_val)


# In[13]:


import matplotlib.pyplot as plt


# In[15]:


plt.plot(Steps, Cost_val)
plt.ylabel('Cost')
plt.xlabel('Step')
plt.ylim(-10, 200)
plt.show()


# In[16]:


print("Your score will be ", sess.run(hypothesis, feed_dict = {X : [[100, 70, 101]]}))


# In[17]:


print("Other score will be ", sess.run(hypothesis, feed_dict = {X : [[60, 70, 110], [90, 100, 80]]}))


# In[ ]:




