#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np


# In[17]:


# from sklearn.datasets import load_breast_cancer # 引入乳腺癌数据集
from sklearn.datasets import load_iris #引入鸢尾花数据集


# In[18]:


# data = load_breast_cancer()
data = load_iris()


# In[19]:


X=data.data # 特征数据
Y=data.target #目标数据


# In[22]:


pd.DataFrame(X).head()


# In[6]:


data.target_names #类别名称


# In[ ]:





# In[7]:


data.feature_names #特征名称


# In[23]:


from sklearn.tree import DecisionTreeClassifier # 引入决策树
from sklearn.model_selection import train_test_split # 引入训练集测试集划分


# In[49]:


DT = DecisionTreeClassifier(criterion='gini',max_depth=5 ,min_samples_split=20) # 实例化

Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size=0.3) #划分数据集

len(Xtrain)

DT.fit(Xtrain,Ytrain) #模型训练

DT.score(Xtest,Ytest) #模型评分

from sklearn import tree 
import graphviz 
info = tree.export_graphviz(DT,feature_names=data.feature_names,class_names=data.target_names,filled=True) #决策树作图

graph = graphviz.Source(info) 
graph #画图


# In[50]:


get_ipython().run_line_magic('pinfo2', 'DecisionTreeClassifier')


# In[ ]:


min_samples_split 


# In[ ]:


min_samples_leaf

