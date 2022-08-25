#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf

# In[3]:


tf.set_random_seed(777)


# In[4]:


x_data = [[73., 80., 75.],
         [93., 88., 93.],
         [89., 91., 90.],
         [96., 98., 100.],
         [73., 66., 70.]]
y_data = [[152.],
         [185.],
         [180.],
         [196.],
         [142.]]


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


optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)


# In[10]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())


# In[15]:


H1 = []
H2 = []
H3 = []
H4 = []
H5 = []
Cost = []


# In[16]:


for step in range(2001):
    cost_val, hy_val, val = sess.run([cost, hypothesis, train], feed_dict = {X : x_data, Y : y_data})
    H1.append(hy_val[0])
    H2.append(hy_val[1])
    H3.append(hy_val[2])
    H4.append(hy_val[3])
    H5.append(hy_val[4])
    Cost.append(cost_val)
    if step % 10 == 0:
        print(step, "Cost : ", cost_val, "\nPrediction : \n", hy_val)


# In[17]:


hy_val.shape


# In[18]:


import matplotlib.pyplot as plt


# In[25]:


plt.plot(H1, Cost, label='H1')
plt.plot(H2, Cost, label='H2')
plt.plot(H3, Cost, label='H3')
plt.plot(H4, Cost, label='H4')
plt.plot(H5, Cost, label='H5')

plt.legend()

plt.show()


# In[ ]:




