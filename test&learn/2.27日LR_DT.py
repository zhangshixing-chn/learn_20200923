#!/usr/bin/env python
# coding: utf-8

# ## Titanic 沉没
# 
# 这是一个分类任务，特征包含离散特征和连续特征，数据如下：[Kaggle地址](https://www.kaggle.com/c/titanic/data)。目标是根据数据特征预测一个人是否能在泰坦尼克的沉没事故中存活下来。接下来解释下数据的格式：
# 
# ```
# survival        目标列，是否存活，1代表存活 (0 = No; 1 = Yes)  
# pclass          乘坐的舱位级别 (1 = 1st; 2 = 2nd; 3 = 3rd)  
# name            姓名 
# sex             性别  
# age             年龄  
# sibsp           兄弟姐妹的数量（乘客中）  
# parch           父母的数量（乘客中）  
# ticket          票号  
# fare            票价  
# cabin           客舱  
# embarked        登船的港口  
#                 (C = Cherbourg; Q = Queenstown; S = Southampton)

# In[1]:


import pandas as pd
import numpy as np
from sklearn import preprocessing # 预处理模块
pd.set_option("display.max_columns",60) #设置展示最大列为60


# ## 导入数据

# In[2]:


train = pd.read_csv("train.csv") #训练数据集 有标签
test = pd.read_csv("test.csv") #测试数据集 无标签
IDtest = test["PassengerId"] #乘客ID


# In[3]:


test.head(2)


# In[4]:


train.head(2)


# In[5]:


train_len = len(train)
dataset =  pd.concat(objs=[train, test], axis=0).reset_index(drop=True)


# In[6]:


dataset.head()


# ## 查看数据

# In[7]:


dataset.tail()


# In[8]:


dataset.dtypes


# In[9]:


dataset.info()


# In[10]:


dataset.isnull().sum()


# ## 特征分析

# In[11]:


# pclass
train.groupby('Pclass')['Survived'].count()


# In[12]:


train.groupby('Pclass')['Survived'].sum()/train.groupby('Pclass')['Survived'].count()


# In[13]:


enc_pclass = preprocessing.OneHotEncoder() # 引入onehot编码类
enc_pclass.fit(dataset[['Pclass']])


# In[14]:


# ['A','B','C'] -> [1,2,3] #LABELENCODE 
# ['A','B','C'] -> [[1,0,0],[0,1,0],[0,0,1]] # onehotencode


# In[15]:


Pclass_feature = pd.DataFrame(enc_pclass.transform(dataset[['Pclass']]).toarray(),
                              columns=['Pclass'+str(i) for i in range(len(dataset['Pclass'].unique()))],dtype=int)


# In[16]:


Pclass_feature.head()


# In[17]:


## name feature analysis
dataset[['Name']].head(5)


# In[18]:


string_ex = 'Braund, Mr. Owen Harris'


# In[19]:


string_ex.split(',')[1].split(".")[0].strip() #切分出所需字段


# In[20]:


# name
dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]] #提取姓名
dataset["Title"] = pd.Series(dataset_title)
dataset["Title"].head()


# In[25]:


dataset.groupby(["Title"])['Name'].count() # 查看数据分布


# In[22]:


dataset["Title"] = dataset["Title"].replace(
    ['Lady', 'the Countess','Countess','Capt', 
     'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')


# In[ ]:





# In[23]:


dataset["Title"] = dataset["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
dataset["Title"] = dataset["Title"].astype(int)


# In[24]:


dataset['Title'].unique()


# In[26]:


enc_name = preprocessing.OneHotEncoder() # 引入onehot编码类
enc_name.fit(dataset[['Title']])


# In[27]:


Name_feature = pd.DataFrame(enc_name.transform(dataset[['Title']]).toarray(),
                              columns=['Title'+str(i) for i in range(len(dataset['Title'].unique()))],dtype=int)


# In[28]:


Name_feature.head()


# In[31]:


dataset.head()


# In[32]:


# sex
train.groupby('Sex')['Survived'].sum()/train.groupby('Sex')['Survived'].count()


# In[33]:


train.groupby('Sex')['Survived'].count()


# In[34]:


## 字符串用数值编码
Sex_feature = pd.DataFrame(dataset['Sex'].map({'female':0,'male':1}).values,columns=['sex'])


# In[35]:


Sex_feature.head()


# In[36]:


dataset[['SibSp','Parch','Pclass']].head(2)


# In[37]:


# Age 填补缺失值
index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index)


# In[38]:


index_NaN_age


# In[39]:


### 利用中位数填充
for i in index_NaN_age :
    age_med = dataset["Age"].median()
    age_pred = dataset["Age"][((dataset['SibSp'] == dataset.iloc[i]["SibSp"]) 
                               & (dataset['Parch'] == dataset.iloc[i]["Parch"]) 
                               & (dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()
    if not np.isnan(age_pred) :
        dataset['Age'].iloc[i] = age_pred
    else :
        dataset['Age'].iloc[i] = age_med
        
### 利用其余特征来构建一个回归，age利用回归给出


# In[ ]:


# 填充思想
#   SibSp Parch Pclass age
#    1     2     2      15
#    1     2     2      16
#    1     2     2      17
#    1     2     3      30
#    2     3     1      16
#


#   1      2     2      nan->median(15,16,17)->16
#   2      5     4      nan->median(15,16,16,17,30)->16


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[40]:


Age_feature = dataset[['Age']]


# In[41]:


Age_feature.head(3)


# In[42]:


# SibSp 特征
dataset['SibSp'].unique()


# In[43]:


dataset.groupby(['SibSp'])['Survived'].count() #查看总量


# In[45]:


dataset.groupby(['SibSp'])['Survived'].sum()/dataset.groupby(['SibSp'])['Survived'].count() #查看存活数量


# In[46]:


# 长尾现象，可以利用分箱的办法处理
SibSp_feature = dataset[['SibSp']]
SibSp_feature['SibSp'] = np.where(SibSp_feature['SibSp']<2,SibSp_feature['SibSp'],2)


# In[47]:


SibSp_feature.SibSp.unique()


# In[48]:


# Parch
train.groupby(['Parch'])['Survived'].count()


# In[49]:


train.groupby(['Parch'])['Survived'].count()


# In[50]:


# 同样的进行分箱处理
Parch_feature = dataset[['Parch']]
Parch_feature['Parch'] = np.where(Parch_feature['Parch']>1,2,Parch_feature['Parch'])


# In[52]:


Parch_feature['Parch'].unique()


# In[303]:


# ticket
dataset['Ticket'].unique()


# In[53]:


Ticket = []
for i in list(dataset.Ticket):
    if not i.isdigit() :
        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0]) #Take prefix
    else:
        Ticket.append("X")


# In[54]:


Ticket_feature = pd.DataFrame(Ticket,columns=['Ticket'])


# In[55]:


Ticket_feature.groupby('Ticket')['Ticket'].count() # 数据偏差较大，不好处理，要么舍弃【大部分人的处理办法】，该特征选择从众原则吧。


# In[56]:


# Fare 
dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())
Fare_feature = dataset[['Fare']]


# In[57]:


Fare_feature.describe()


# In[ ]:





# In[58]:


# Embarked
dataset["Embarked"].isnull().sum()


# In[59]:


dataset.groupby(["Embarked"])['Fare'].count()


# In[60]:


dataset["Embarked"] = dataset["Embarked"].fillna("S") # 众数填充Embarked


# In[62]:


enc_Embarked = preprocessing.OneHotEncoder()
enc_Embarked.fit(dataset[["Embarked"]])


# In[63]:


Embarked_feature = pd.DataFrame(enc_Embarked.transform(dataset[['Embarked']]).toarray(),
            columns=['Embarked'+str(i) for i in range(len(dataset['Embarked'].unique()))],dtype=int)


# In[64]:


Embarked_feature.head()


# In[65]:


# Cabin
dataset["Cabin"].head()


# In[66]:


dataset["Cabin"].unique()


# In[67]:


Cabin_feature = dataset[["Cabin"]] # 取出Cabin


# In[68]:


Cabin_feature = Cabin_feature.fillna('X') ## 空值填充为X


# In[69]:


Cabin_feature['Cabin'] = Cabin_feature['Cabin'].map(lambda x: x[0]) ## 取出Cabin的首个字符


# In[70]:


Cabin_feature.groupby(['Cabin'])['Cabin'].count()


# In[72]:


Cabin_feature[Cabin_feature['Cabin'].isin(['G','T'])]='F'


# In[73]:


enc_Cabin = preprocessing.OneHotEncoder() # 引入onehot编码类
enc_Cabin.fit(Cabin_feature[['Cabin']])
Cabin_feature = pd.DataFrame(enc_Cabin.transform(Cabin_feature[['Cabin']]).toarray(),
                             columns=['Cabin'+str(i) for i in range(len(Cabin_feature['Cabin'].unique()))],dtype=int)


# In[74]:


Cabin_feature.head()


# In[76]:


#### new feature
# family size
dataset["Fsize"] = dataset["SibSp"] + dataset["Parch"] + 1


# In[77]:


dataset.groupby(['Fsize'])['Survived'].count()


# In[78]:


dataset.groupby(['Fsize'])['Survived'].sum()/dataset.groupby(['Fsize'])['Survived'].count()


# In[79]:


Fsize_feature = dataset[['Fsize']]


# In[80]:


## 大于5的替换为5
Fsize_feature['Fsize'] = np.where(Fsize_feature['Fsize']>=5,5,Fsize_feature['Fsize']) 


# In[81]:


## 2-5之间替换为2
Fsize_feature['Fsize'] = np.where((Fsize_feature['Fsize']<5)&(Fsize_feature['Fsize']>=2),2,Fsize_feature['Fsize'])


# In[82]:


Fsize_feature['Fsize'].unique()


# In[ ]:





# In[83]:


## 整合所有特征
feature_list = ['Name_feature','Sex_feature','Age_feature','SibSp_feature','Parch_feature',
'Fare_feature','Embarked_feature','Cabin_feature','Fsize_feature']
baseinfo = dataset[['PassengerId','Survived']]
for iname in feature_list:
    baseinfo = pd.concat([baseinfo,eval(iname)],axis=1)


# In[84]:


baseinfo.head()


# In[85]:


baseinfo.shape


# In[ ]:





# In[86]:


train_df = baseinfo[baseinfo['PassengerId'].isin(list(train['PassengerId']))] ### 训练集的id


# In[87]:


test_df = baseinfo[baseinfo['PassengerId'].isin(list(test['PassengerId']))] ### 测试集的id


# In[88]:


train_df.head()


# In[89]:


test_df.head()


# In[338]:


## 构建模型决策树 和 逻辑回归


# In[92]:


from sklearn.model_selection import train_test_split # 训练集和测试集划分
from sklearn.tree import DecisionTreeClassifier # 引入分类树模型
from sklearn.linear_model import LogisticRegression # 引入逻辑回归模型
from sklearn.model_selection import GridSearchCV # 格点搜索调参


# In[93]:


Dtrain = train_df.iloc[0::,2::]
y_trains = train_df[['Survived']]
Dtest = test_df.iloc[0::,2::]


# In[94]:


train_df.head()


# In[95]:


Xtrain,Xvalid,ytrain,yvalid = train_test_split(Dtrain,y_trains,test_size=0.3, random_state=42)


# In[96]:


Dtrain.head()


# In[97]:


### 利用格点搜索的方式选取最优的参数
param_test1 = {'max_depth':range(3,10,1),'criterion':['entropy','gini']}
gsearch1 = GridSearchCV(estimator = DecisionTreeClassifier(),param_grid = param_test1, 
                        scoring='roc_auc', cv=5)


# In[98]:


gsearch1.fit(Xtrain,ytrain)


# In[99]:


gsearch1.scorer_, gsearch1.best_params_, gsearch1.best_score_ 


# In[100]:


DT = DecisionTreeClassifier(criterion='gini',max_depth=4)


# In[101]:


DT.fit(Xtrain,ytrain)


# In[102]:


DT.score(Xvalid,yvalid)


# In[103]:


DT.fit(Dtrain,y_trains)


# In[104]:


DT.feature_importances_


# In[105]:


### 真实的预测

test_Survived = pd.Series(DT.predict(Dtest), name="Survived")

results = pd.concat([IDtest,test_Survived],axis=1)

results['Survived'] = np.where(results['Survived']>=0.5,1,0)

results.to_csv("ensemble_python_vertwo.csv",index=False)


# In[106]:


results.head()


# In[107]:


#### 逻辑回归


# In[108]:


train_df.head()


# In[123]:


Age_enc = preprocessing.StandardScaler()
train_df['Age'] = Age_enc.fit_transform(train_df[['Age']]) # 训练数据标准化
test_df['Age'] = Age_enc.transform(test_df[['Age']]) #测试数据标准化


# In[124]:


Fare_enc = preprocessing.StandardScaler()
train_df['Fare'] = Fare_enc.fit_transform(train_df[['Fare']])
test_df['Fare'] = Fare_enc.transform(test_df[['Fare']])


# In[125]:


#  1，2，3 -> fit_transform -> -1, 0, 1 (x-mean) 均值：2


# 1,1,1 -> transform -> -1,-1,-1  均值：2


#  1，1，1 -> fit_transform -> 0,0,0  均值：1 


# In[ ]:





# In[111]:


train_df.head()


# In[112]:


Dtrain = train_df.iloc[0::,2::]
y_trains = train_df[['Survived']]
Dtest = test_df.iloc[0::,2::]


# In[113]:


Xtrain,Xvalid,ytrain,yvalid = train_test_split(Dtrain,y_trains,test_size=0.3, random_state=42)


# In[115]:


### 利用格点搜索的方式选取最优的参数
import warnings
warnings.filterwarnings("ignore")


# In[116]:


param_test1 = {'C':[0.01,0.1,1,10,100],'penalty':['l1','l2']}
gsearch1 = GridSearchCV(estimator = LogisticRegression(),param_grid = param_test1, 
                        scoring='roc_auc', cv=5)
gsearch1.fit(Xtrain,ytrain)


# In[117]:


gsearch1.scorer_, gsearch1.best_params_, gsearch1.best_score_ 


# In[118]:


LR  = LogisticRegression(C=1,penalty='l1')


# In[119]:


LR.fit(Dtrain,y_trains)


# In[120]:


LR.coef_


# In[364]:


######### 输出真正的test数据标签


# In[365]:


### LR

test_Survived = pd.Series(LR.predict(Dtest), name="Survived")

results = pd.concat([IDtest,test_Survived],axis=1)

results['Survived'] = np.where(results['Survived']>=0.5,1,0)

results.to_csv("ensemble_python_vertwo.csv",index=False)


# In[121]:


results.head()


# In[122]:


# 1. 查看缺失值,填充或者丢弃。
# 2. 对于特征的再挖掘，尤其是文本类型的特征。
# 3. 噪声较大的特征进行处理，删除或者再挖掘。
# 4. 特征的onehot编码和分箱操作。
# 5. 新特征的构造，特征组合。


# 选择模型：
# 1. 树模型不需要标准化和归一化处理。
# 2. 线性模型，需要标准化或者归一化。

# 调参
# 1. 选择格点搜索来进行调参。


# In[367]:


Dtest.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### 构建模型1 XGBOOST

# In[76]:


import xgboost as xgb
from sklearn.model_selection import train_test_split


# In[107]:


params={
    'eta': 0.3,
    'max_depth':5,   
    'min_child_weight':1,
    'gamma':0.1, 
    'subsample':0.8,
    'colsample_bytree':0.8,
    'booster':'gbtree',
    'objective': 'binary:logistic',
    'scale_pos_weight': 1,
    'silent':0 ,
    'eval_metric': 'auc'
}


# In[110]:


kk = xgb.cv(params, d_train, nfold=3, metrics={'error'},early_stopping_rounds = 5, seed=42)


# In[115]:


xg


# In[ ]:





# In[81]:


Dtrain = train_df.iloc[0::,2::]
y_train = train_df[['Survived']]
Dtest = test_df.iloc[0::,2::]


# In[82]:


Xtrain,Xvalid,ytrain,yvalid = train_test_split(Dtrain,y_train,test_size=0.3, random_state=42)
Xtrain = Xtrain.reset_index(drop=True)
Xvalid = Xvalid.reset_index(drop=True)
ytrain = ytrain.reset_index(drop=True)
yvalid = yvalid.reset_index(drop=True)


# In[83]:


d_train = xgb.DMatrix(Xtrain, label=ytrain)
d_valid = xgb.DMatrix(Xvalid, label=yvalid)
all_train = xgb.DMatrix(Dtrain,label=y_train)
test = xgb.DMatrix(Dtest)


# In[84]:


watchlist = [(d_train, 'train'), (d_valid, 'valid')]


# In[114]:


model_bst = xgb.train(params, d_train, 50, watchlist, early_stopping_rounds=30, verbose_eval=1)


# In[118]:


params={
    'eta': 0.3,
    'max_depth':3,   
    'min_child_weight':1,
    'gamma':0.1, 
    'subsample':0.8,
    'colsample_bytree':0.8,
    'booster':'gbtree',
    'objective': 'binary:logistic',
    'scale_pos_weight': 1,
    'silent':0     
}


# In[119]:


model_bst = xgb.train(params, all_train, 9)


# In[285]:


# predict

test_Survived = pd.Series(model_bst.predict(test), name="Survived")

results = pd.concat([IDtest,test_Survived],axis=1)

results['Survived'] = np.where(results['Survived']>=0.5,1,0)

results.to_csv("ensemble_python_verone.csv",index=False)


# ### 构建模型2  XGBOOST +LR
# 

# In[120]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBClassifier


# In[121]:


Dtrain = train_df.iloc[0::,2::].reset_index(drop=True)
y_train = train_df[['Survived']].reset_index(drop=True)
Dtest = test_df.iloc[0::,2::].reset_index(drop=True)


# In[125]:


param_test1 = {
    'n_estimators':range(3,40,2)}
gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.3, max_depth=3,
                                        min_child_weight=1, gamma=0.1, subsample=0.8, colsample_bytree=0.8,
                                        objective= 'binary:logistic', nthread=4, scale_pos_weight=1,random_state=42), 
param_grid = param_test1, scoring='roc_auc',n_jobs=4, cv=5)
gsearch1.fit(Dtrain,y_train)


# In[126]:


gsearch1.scorer_, gsearch1.best_params_, gsearch1.best_score_


# In[127]:


best_paras = {
    'n_estimators': 11,
    'learning_rate': 0.3,
    'max_depth':3,   
    'min_child_weight':1,
    'gamma':0.1, 
    'subsample':0.8,
    'colsample_bytree':0.8,
    'booster':'gbtree',
    'objective': 'binary:logistic',
    'scale_pos_weight': 1,
    'random_state':42}


# In[128]:


layer0_estimator = XGBClassifier(**best_paras)


# In[129]:


layer0_estimator.fit(Dtrain,y_train)


# In[130]:


X_train_leaves = layer0_estimator.apply(Dtrain)


# In[132]:


X_train_leaves


# In[133]:


X_train_leaves.shape


# In[131]:


X_test_leaves = layer0_estimator.apply(Dtest)


# In[134]:


xgbenc = preprocessing.OneHotEncoder()
X_trans = xgbenc.fit_transform(X_train_leaves)
X_test = xgbenc.transform(X_test_leaves)


# In[135]:


X_trans


# In[ ]:





# In[136]:


# new feature
Dtrain_new = pd.concat([pd.DataFrame(X_trans.toarray()),Dtrain],axis=1)
Dtest_new = pd.concat([pd.DataFrame(X_test.toarray()),Dtest],axis=1)


# In[137]:


Dtrain_new.shape


# In[138]:


# 引入逻辑回归
lr = LogisticRegression(penalty='l2')


# In[139]:


param_test1 = {
    'C':range(1,100,10)}
gsearch1 = GridSearchCV(estimator = LogisticRegression(penalty='l2',random_state=42), 
param_grid = param_test1, scoring='roc_auc',n_jobs=4, cv=5)
gsearch1.fit(Dtrain_new,y_train)


# In[140]:


gsearch1.scorer_, gsearch1.best_params_, gsearch1.best_score_


# In[142]:


best_paras = {'C':1,'penalty':'l2'}


# In[143]:


lr = LogisticRegression(**best_paras)


# In[144]:


lr.fit(Dtrain_new,y_train)


# In[397]:


# predict

test_Survived = pd.Series(lr.predict(Dtest_new), name="Survived")

results = pd.concat([IDtest,test_Survived],axis=1)

results['Survived'] = np.where(results['Survived']>=0.5,1,0)

results.to_csv("ensemble_python_vertwo.csv",index=False)


# In[ ]:




