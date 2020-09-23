#!/usr/bin/env python
# coding: utf-8

# 学习中的回归问题，咱们一起来看看这个问题。<br><br>
# 这是一个城市自行车租赁系统，提供的数据为2年内华盛顿按小时记录的自行车租赁数据，其中训练集由每个月的前19天组成，测试集由20号之后的时间组成（需要我们自己去预测）

# In[6]:


# 环境 python3.7
# sklearn 0.20


# In[13]:


import sklearn


# In[14]:


sklearn.__version__


# In[15]:


import numpy as np
import pandas as pd


# In[16]:


df_train = pd.read_csv('train.csv')


# In[18]:


df_test = pd.read_csv('test.csv')


# In[19]:


df_test.head()


# In[17]:


df_train.head() #查看头5行


# In[7]:


df_train.dtypes #查看字段的名字和类型


# datetime       object 日期<br>
# season          int64 季节<br>
# holiday         int64 是否假期<br>
# workingday      int64 是否工作日<br>
# weather         int64 天气<br>
# temp          float64 温度<br>
# atemp         float64 体感温度<br>
# humidity        int64 湿度<br>
# windspeed     float64 风速<br>
# casual          int64 未注册用户租车数量<br>
# registered      int64 注册用户租车数量<br>
# count           int64 总数量<br>

# In[8]:


df_train.shape #数据量大小


# In[9]:


df_train.describe() # 数据的基本统计量


# In[20]:


df_train.info() #数据的缺省等信息 如果有缺失值，需要对缺失值进行处理


# In[22]:


## 对日期进行处理
df_train['month'] = pd.DatetimeIndex(df_train.datetime).month
df_train['day'] = pd.DatetimeIndex(df_train.datetime).dayofweek
df_train['hour'] = pd.DatetimeIndex(df_train.datetime).hour


# In[23]:


df_train.head()


# In[24]:


df_train = df_train.drop(['datetime'], axis = 1) #删除处理过的时间数据


# In[26]:


df_train.drop(['casual','registered'], axis=1, inplace=True) # 删除未注册用户和注册用户，因为总体用户是由这两个用户相加得来


# In[27]:


df_train.head()


# In[28]:


# 来一个粗略的版本
# 确定特征和标签
df_train_target = df_train['count'].values
df_train_data = df_train.drop(['count'],axis = 1).values


# In[29]:


## 机器学习算法
from sklearn.model_selection import train_test_split


# In[31]:


## 划分数据集
X_train, X_test, y_train, y_test = train_test_split(df_train_data, df_train_target, test_size=0.2, random_state=1)


# In[34]:


from sklearn.linear_model import RidgeCV
from sklearn import metrics
alphas = 10**np.linspace(-5, 5, 50)
rd_cv = RidgeCV(alphas=alphas, cv=5, scoring='r2')
rd_cv.fit(X_train, y_train)
rd_cv.alpha_


# In[35]:


from sklearn.linear_model import Ridge
rd = Ridge(alpha=2329.951810515372) #, fit_intercept=False
rd.fit(X_train, y_train)


# In[36]:


from sklearn import metrics
from math import sqrt
#分别预测训练数据和测试数据
y_train_pred = rd.predict(X_train)
y_test_pred = rd.predict(X_test)


# In[38]:


y_test_pred


# In[37]:


y_train_pred


# In[40]:


#计算其均方根误差
y_train_pred[y_train_pred<=0]=0.01
y_test_pred[y_test_pred<=0]=0.01


# In[41]:


y_train_rmse = sqrt(metrics.mean_squared_error(np.log(y_train), np.log(y_train_pred)))
y_test_rmse = sqrt(metrics.mean_squared_error(np.log(y_test), np.log(y_test_pred)))
print('训练集RMSE: {0}'.format(y_train_rmse))
print('测试集RMSE: {0}'.format(y_test_rmse))


# In[42]:


print(rd.coef_)
print(rd.intercept_)


# In[23]:


y_train_pred


# In[24]:


y_train


# In[43]:


########## LASSO
from sklearn.linear_model import LassoCV
from sklearn import metrics
Las_cv = LassoCV(alphas=alphas, cv=10)
Las_cv.fit(X_train, y_train)
Las_cv.alpha_


# In[44]:


from sklearn.linear_model import Lasso
Las = Lasso(alpha=1.2648) #, fit_intercept=False
Las.fit(X_train, y_train)


# In[70]:


Las.coef_


# In[45]:


y_train_pred = Las.predict(X_train)
y_test_pred = Las.predict(X_test)
y_train_pred[y_train_pred<=0]=0.01
y_test_pred[y_test_pred<=0]=0.01

y_train_rmse = sqrt(metrics.mean_squared_error(np.log(y_train), np.log(y_train_pred)))
y_test_rmse = sqrt(metrics.mean_squared_error(np.log(y_test), np.log(y_test_pred)))
print('训练集RMSE: {0}'.format(y_train_rmse))
print('测试集RMSE: {0}'.format(y_test_rmse))


# In[47]:


10**1.7464


# In[48]:


df_train['count'].mean()


# In[49]:


print(Las.coef_)
print(Las.intercept_)


# In[ ]:





# In[43]:


y_test_pred


# In[49]:


####### linear regression 
from sklearn.linear_model import LinearRegression
#训练线性回归模型
LR = LinearRegression()
LR.fit(X_train, y_train)
print(LR.coef_)
print(LR.intercept_)
#分别预测训练集和测试集, 并计算均方根误差
y_train_pred = LR.predict(X_train)
y_test_pred = LR.predict(X_test)
y_train_pred[y_train_pred<=0]=0.01
y_test_pred[y_test_pred<=0]=0.01

y_train_rmse = sqrt(metrics.mean_squared_error(np.log(y_train), np.log(y_train_pred)))
y_test_rmse = sqrt(metrics.mean_squared_error(np.log(y_test), np.log(y_test_pred)))
print('训练集RMSE: {0}'.format(y_train_rmse))
print('测试集RMSE: {0}'.format(y_test_rmse))


# In[ ]:





# # 做点特征工程吧

# In[50]:


import numpy as np
import pandas as pd
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')


# In[51]:


df_train.head()


# In[84]:


df_test.head()


# In[52]:


## 对日期进行处理
df_train['month'] = pd.DatetimeIndex(df_train.datetime).month
df_train['day'] = pd.DatetimeIndex(df_train.datetime).dayofweek
df_train['hour'] = pd.DatetimeIndex(df_train.datetime).hour


# In[53]:


df_train.head()


# In[54]:


df_train['season'].unique()


# In[55]:


df_train.groupby(['season'])['count'].count()


# In[ ]:


# 1 - [1,0,0,0]
# 2 - [0,1,0,0]


# In[56]:


from sklearn import preprocessing


# In[57]:


enc_season = preprocessing.OneHotEncoder()
enc_season.fit(df_train[['season']]) # 建立标准


# In[60]:


Season_feature = pd.DataFrame(enc_season.transform(df_train[['season']]).toarray(),
                              columns=['season'+str(i) for i in range(len(df_train['season'].unique()))],dtype=int)


# In[67]:


enc_season.transform([[4]]).toarray()


# In[68]:


Season_feature.head()


# In[5]:


enc_season.inverse_transform(Season_feature) # 逆解析


# In[69]:


## holiday
df_train['holiday'].unique()


# In[ ]:





# In[45]:


df_train.groupby(['holiday'])['count'].count()


# In[71]:


df_train['workingday'].unique()


# In[46]:


df_train.groupby(['workingday'])['count'].count()


# In[72]:


df_train['weather'].unique()


# In[48]:


df_train.groupby(['weather'])['count'].count()


# In[51]:


### weather 反映了天气的恶劣程度，因此不做onehot编码


# In[74]:


# 温度
import matplotlib.pyplot as plt
df_train.groupby('temp').mean().plot(y='count', marker='o')
plt.show()


# In[75]:


plt.hist(df_train['temp'])
plt.show()


# In[77]:


from sklearn.preprocessing import StandardScaler
enc_temp = StandardScaler()


# In[78]:


temp_feature = pd.DataFrame(enc_temp.fit_transform(df_train[['temp']]),columns=['temp'])


# In[80]:


temp_feature.head()


# In[ ]:


# fit-transform


# In[99]:


# 湿度
df_train.groupby('atemp').mean().plot(y='count', marker='o')
plt.show()


# In[100]:


plt.hist(df_train['atemp'])
plt.show()


# In[101]:


enc_atemp = StandardScaler()
atemp_feature = pd.DataFrame(enc_atemp.fit_transform(df_train[['atemp']]),columns=['atemp'])


# In[82]:


# 湿度
df_train.groupby('humidity').mean().plot(y='count', marker='o')
plt.show()


# In[83]:


plt.hist(df_train['humidity'])
plt.show()


# In[84]:


enc_hum = StandardScaler()
humidity_feature = pd.DataFrame(enc_hum.fit_transform(df_train[['humidity']]),columns=['humidity'])


# In[85]:


# 风速
df_train.groupby('windspeed').mean().plot(y='count', marker='o')
plt.show()


# In[86]:


plt.hist(df_train['windspeed'])
plt.show()


# In[87]:


enc_speed = StandardScaler()
windspeed_feature = pd.DataFrame(enc_speed.fit_transform(df_train[['windspeed']]),columns=['windspeed'])


# In[88]:


plt.hist(df_train['count'])
plt.show()


# In[89]:


count_target = pd.DataFrame(np.log(df_train['count']),columns=['count'])


# In[90]:


plt.hist(count_target['count'])
plt.show()


# In[121]:


## month hour week


# In[131]:


df_train['hour'].unique()


# In[96]:


# month
enc_month = preprocessing.OneHotEncoder()
enc_month.fit(df_train[['month']])
month_feature = pd.DataFrame(enc_month.transform(df_train[['month']]).toarray(),
                              columns={'month'+str(i) for i in range(len(df_train['month'].unique()))},dtype=int)


# In[134]:


month_feature.head()


# In[97]:


# day
enc_day = preprocessing.OneHotEncoder()
enc_day.fit(df_train[['day']])
day_feature = pd.DataFrame(enc_day.transform(df_train[['day']]).toarray(),
                              columns=['day'+str(i) for i in range(len(df_train['day'].unique()))],dtype=int)


# In[137]:


day_feature.head()


# In[91]:


## hour 
est = preprocessing.KBinsDiscretizer(n_bins=[6])
est.fit(df_train[['hour']])


# In[94]:


hour_feature = pd.DataFrame(est.transform(df_train[['hour']]).toarray(),
                              columns=['hour'+str(i) for i in range(6)],dtype=int)


# In[95]:


hour_feature.head()


# In[ ]:





# In[102]:


df_train_data = pd.concat([Season_feature,df_train[['holiday']],df_train[['workingday']],
                          df_train[['weather']],temp_feature,atemp_feature,humidity_feature,
                          windspeed_feature,month_feature,day_feature,hour_feature],axis=1)


# In[103]:


df_train_target = count_target


# In[104]:


df_train_data.shape


# In[110]:


df_train_data.head()


# In[ ]:





# In[105]:


## 划分数据集
X_train, X_test, y_train, y_test = train_test_split(df_train_data, df_train_target, test_size=0.2, random_state=1)


# In[106]:


from sklearn.linear_model import RidgeCV
from sklearn import metrics
alphas = 10**np.linspace(-5, 5, 50)
rd_cv = RidgeCV(alphas=alphas, cv=5, scoring='r2')
rd_cv.fit(X_train, y_train)
rd_cv.alpha_


# In[107]:


from sklearn.linear_model import Ridge
rd = Ridge(alpha=5.1794746792312125) #, fit_intercept=False
rd.fit(X_train, y_train)


# In[108]:


from sklearn import metrics
from math import sqrt
#分别预测训练数据和测试数据
y_train_pred = rd.predict(X_train)
y_test_pred = rd.predict(X_test)
#计算其均方根误差


y_train_rmse = sqrt(metrics.mean_squared_error(y_train,y_train_pred))
y_test_rmse = sqrt(metrics.mean_squared_error(y_test,y_test_pred))
print('训练集RMSE: {0}'.format(y_train_rmse))
print('测试集RMSE: {0}'.format(y_test_rmse))


# In[109]:


10**0.9


# In[172]:


print(rd.coef_)
print(rd.intercept_)


# In[174]:


########## LASSO
from sklearn.linear_model import LassoCV
from sklearn import metrics
Las_cv = LassoCV(alphas=alphas, cv=10)
Las_cv.fit(X_train, y_train)
Las_cv.alpha_


# In[175]:


from sklearn.linear_model import Lasso
Las = Lasso(alpha=0.0002682695795279727) #, fit_intercept=False
Las.fit(X_train, y_train)


# In[176]:


y_train_pred = Las.predict(X_train)
y_test_pred = Las.predict(X_test)

y_train_rmse = sqrt(metrics.mean_squared_error(y_train,y_train_pred))
y_test_rmse = sqrt(metrics.mean_squared_error(y_test,y_test_pred))
print('训练集RMSE: {0}'.format(y_train_rmse))
print('测试集RMSE: {0}'.format(y_test_rmse))


# In[177]:


print(Las.coef_)
print(Las.intercept_)


# In[ ]:





# In[178]:


####### linear regression 
from sklearn.linear_model import LinearRegression
#训练线性回归模型
LR = LinearRegression()
LR.fit(X_train, y_train)
print(LR.coef_)
print(LR.intercept_)
#分别预测训练集和测试集, 并计算均方根误差
y_train_pred = LR.predict(X_train)
y_test_pred = LR.predict(X_test)


y_train_rmse = sqrt(metrics.mean_squared_error(y_train,y_train_pred))
y_test_rmse = sqrt(metrics.mean_squared_error(y_test,y_test_pred))
print('训练集RMSE: {0}'.format(y_train_rmse))
print('测试集RMSE: {0}'.format(y_test_rmse))


# In[ ]:


# LR， encoder


# In[ ]:





# In[ ]:




