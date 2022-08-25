#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf

# In[3]:


hello = tf.constant("Hello World")


# In[4]:


print(hello)


# In[5]:


a = tf.constant(10)
b = tf.constant(32)
c = a + b


# In[6]:


print(c)


# In[7]:


sess = tf.Session()


# In[8]:


print(sess.run(hello))


# In[10]:


print(str(sess.run(hello), encoding="utf-8"))


# In[13]:


print(sess.run([a, b, c]))


# In[14]:


sess.close()


# In[15]:


node1 = tf.constant(3.0)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)


# In[16]:


with tf.Session() as sess:
    print("node3 : ", node3)
    print("sess.run(node3) : ", sess.run(node3))


# Node 1, 2 를 실행하지 않아도 그래프로 연결되어있기 때문에 Node 1, 2가 더해져서 Node 3 이 출력된다.

# In[17]:


input01 = tf.placeholder(tf.float32)
input02 = tf.placeholder(tf.float32)
output = tf.multiply(input01, input02)


# In[18]:


print("input01 : ", input01)
print("input02 : ", input02)


# In[20]:


with tf.Session() as sess:
    print(sess.run(output, feed_dict={input01 : 3.0, input02 : 5.0}))
    print(sess.run(output, feed_dict={input01 : 0.0, input02 : 6.0}))
    print(sess.run(output, feed_dict={input01 : [2.0], input02 : [6.0]}))


# In[21]:


X = tf.placeholder(tf.float32, [None, 3])
#None은 크기가 정해지지 않았음을 의미한다. (얼마나 들어올 지 모르겠다!)


# In[23]:


print(X)


# In[24]:


x_data = [[1, 2, 3], [4, 5, 6]]
#X placeholder에 넣을 값
#두 번째 차원의 요소의 개수는 3


# In[25]:


w = tf.Variable(tf.random_normal([3, 2]))
b = tf.Variable(tf.random_normal([2, 1]))
#tf.Variable() : 그래프를 계산하면서 최적화 할 함수
#tf.random_normal() : 각 변수의 초기 값을 정규분포 랜덤 값으로 초기화


# In[27]:


expr = tf.matmul(X, w) + b
#입력값과 변수들을 계산할 수식을 작성
#tf.matmul 처럼 mat로 시작하는 함수로 행렬계산을 수행 (행렬 곱셈)


# In[28]:


sess = tf.Session()


# In[29]:


sess.run(tf.global_variables_initializer())


# In[30]:


print(x_data)


# In[31]:


print(sess.run(w))


# In[32]:


print(sess.run(b))


# In[33]:


print(sess.run(expr, feed_dict={X : x_data}))


# In[34]:


sess.close()


# In[ ]:




