#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.display import display
import pandas as pd

# In[2]:


data = {'Name' : ["John", "Anna", "Peter", "Linda"], 'Location' : ["New York", "Paris", "Berlin", "London"], 'Age' : [24, 13, 53, 33]}


# In[3]:


data_pandas = pd.DataFrame(data)


# In[4]:


display(data_pandas)


# In[5]:


my_series = pd.Series({"United Kingdom":"London", "India":"New Delhi", "United States":"Washington", "Belgium":"Brussels"})


# In[6]:


display(pd.DataFrame(my_series))


# In[7]:


my_df = pd.DataFrame(data = [4, 5, 6, 7], index=range(0, 4), columns=['A'])


# In[8]:


display(pd.DataFrame(my_df))


# In[ ]:




