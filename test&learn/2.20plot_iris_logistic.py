#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets


# In[3]:


# import some data to play with
iris = datasets.load_iris()


# In[4]:


iris


# In[5]:


X = iris.data[:, :2]  # we only take the first two features.
Y = iris.target


# In[6]:


logreg = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial',class_weight={1: 2})


# In[7]:


get_ipython().run_line_magic('pinfo', 'LogisticRegression')


# In[8]:


logreg.fit(X, Y)


# In[9]:


# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
h = .02  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])


# In[10]:





# In[11]:


# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()


# In[8]:


iris.target


# In[9]:


Z


# In[10]:


logreg.predict_proba(np.c_[xx.ravel(), yy.ravel()])


# In[11]:


get_ipython().run_line_magic('pinfo2', 'LogisticRegression')


# In[20]:


logreg.coef_


# In[ ]:




