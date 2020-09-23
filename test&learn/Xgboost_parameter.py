
# xgboost 安装
# xgboost是一个独立的，开源的机器学习库
# pip install xgboost 安装原生库
###  import xgboost as xgb #原生的模块 
### 最核心的，是DMtarix这个读取数据类

# sklearn 中的api
###  from xgboost.sklearn import XGBClassifier # 封装到sklearn的api

# 参数说明：
# 1. n_estimator: 也作num_boosting_rounds
# 这是生成的最大树的数目，也是最大的迭代次数。
# 2. learning_rate: 有时也叫作eta，系统默认值为0.3,。
# 每一步迭代的步长，很重要。太大了运行准确率不高，太小了运行速度慢。我们一般使用比默认值小一点，0.1左右就很好。
# 3.subsample：系统默认为1。
# 这个参数控制对于每棵树，随机采样的比例。减小这个参数的值，算法会更加保守，避免过拟合。
# 但是，如果这个值设置得过小，它可能会导致欠拟合。 典型值：0.5-1，0.5代表平均采样，防止过拟合. 范围: (0,1]，注意不可取0
# 4.colsample_bytree：系统默认值为1。我们一般设置成0.8左右。
# 用来控制每棵随机采样的列数的占比(每一列是一个特征)。 典型值：0.5-1范围: (0,1]
# 5.max_depth： 系统默认值为6
# 我们常用3-10之间的数字。这个值为树的最大深度。这个值是用来控制过拟合的。max_depth越大，模型学习的更加具体。
# 设置为0代表没有限制，范围: [0,∞]
# 6. lambda:也称reg_lambda,默认值为0。
# 权重的L2正则化项。这个参数是用来控制XGBoost的正则化部分的。这个参数在减少过拟合上很有帮助。
# 7. alpha:也称reg_alpha默认为0,
# 权重的L1正则化项。(和Lasso regression类似)。 可以应用在很高维度的情况下，使得算法的速度更快。
# 8. gamma：系统默认为0,我们也常用0。
# 在节点分裂时，只有分裂后损失函数的值下降了，才会分裂这个节点。
# gamma指定了节点分裂所需的最小损失函数下降值。 
# 这个参数的值越大，算法越保守。
## xgb --> XGBClassifier
# 1、eta -> learning_rate
# 2、lambda -> reg_lambda
# 3、alpha -> reg_alpha
# 4、num_boosting_rounds -> n_estimators
import pandas as pd
import numpy as np  
import xgboost as xgb #原生的模块 
from xgboost.sklearn import XGBClassifier # 封装到sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

X, y = make_classification(n_samples=800)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

de_param={
'n_estimators':100,
'max_depth':10,
'min_child_weight':5,
'gamma':0,
'subsample':0.2,
'colsample_bytree':0.3,
'reg_alpha':0,
'learning_rate':0.1,
'random_state':10}

rf0 = xgb.XGBClassifier(**de_param) #注意两个**加上de_param,否则报错
rf0.fit(X_train,y_train)
y_trainprob = rf0.predict_proba(X_train)[:,1]
y_testprob = rf0.predict_proba(X_test)[:,1]
print ("AUC Score (Train): %f" % roc_auc_score(y_train, y_trainprob))
print ("AUC Score (Test): %f" % roc_auc_score(y_test, y_testprob))

#get_ipython().run_line_magic('pinfo', 'rf0')

# n_estimators=100,
# max_depth=3,
# gamma=0,
# subsample=0.8,
# colsample_bytree=0.9,
# reg_alpha=0,
# learning_rate =0.1,
# random_state=10

param_test1 = {'n_estimators':range(10,300,20),'learning_rate':[0.01,0.05,0.1,0.2,0.5]} # 调节n_estimators参数
model=xgb.XGBClassifier(**de_param) # 实例化模型
gsearch1 = GridSearchCV(estimator = model,param_grid = param_test1, scoring='roc_auc',cv=3)
gsearch1.fit(X,y)
gsearch1.best_estimator_, gsearch1.best_params_, gsearch1.best_score_

de_param['n_estimators']=50
de_param['learning_rate']=0.1

rf0 = xgb.XGBClassifier(**de_param)
rf0.fit(X_train,y_train)
y_trainprob = rf0.predict_proba(X_train)[:,1]
y_testprob = rf0.predict_proba(X_test)[:,1]
print ("AUC Score (Train): %f" % roc_auc_score(y_train, y_trainprob))
print ("AUC Score (Test): %f" % roc_auc_score(y_test, y_testprob))

param_test1 = {'max_depth':range(1,7,1)} #树最大深度，不能太深，太深的话，存在过拟合的风险
model=xgb.XGBClassifier(**de_param)
gsearch1 = GridSearchCV(estimator = model,param_grid = param_test1, scoring='roc_auc',cv=3)

gsearch1.fit(X,y)
gsearch1.best_estimator_, gsearch1.best_params_, gsearch1.best_score_

de_param['max_depth']=3

rf0 = xgb.XGBClassifier(**de_param)
rf0.fit(X_train,y_train)
y_trainprob = rf0.predict_proba(X_train)[:,1]
y_testprob = rf0.predict_proba(X_test)[:,1]
print ("AUC Score (Train): %f" % roc_auc_score(y_train, y_trainprob))
print ("AUC Score (Test): %f" % roc_auc_score(y_test, y_testprob))

param_test1 = {'gamma':[0,0.01,0.1,10]} 
model=xgb.XGBClassifier(**de_param)
gsearch1 = GridSearchCV(estimator = model,param_grid = param_test1, scoring='roc_auc',cv=3)

gsearch1.fit(X,y)
gsearch1.best_estimator_, gsearch1.best_params_, gsearch1.best_score_

de_param['gamma']=0

rf0 = xgb.XGBClassifier(**de_param)
rf0.fit(X_train,y_train)
y_trainprob = rf0.predict_proba(X_train)[:,1]
y_testprob = rf0.predict_proba(X_test)[:,1]
print ("AUC Score (Train): %f" % roc_auc_score(y_train, y_trainprob))
print ("AUC Score (Test): %f" % roc_auc_score(y_test, y_testprob))

param_test1 = {'subsample':[0.2,0.4,0.8,1.0]} #调节子样本的比率
model=xgb.XGBClassifier(**de_param)
gsearch1 = GridSearchCV(estimator = model,param_grid = param_test1, scoring='roc_auc',cv=3)

gsearch1.fit(X,y)
gsearch1.best_estimator_, gsearch1.best_params_, gsearch1.best_score_

de_param['subsample']=0.8

rf0 = xgb.XGBClassifier(**de_param)
rf0.fit(X_train,y_train)
y_trainprob = rf0.predict_proba(X_train)[:,1]
y_testprob = rf0.predict_proba(X_test)[:,1]
print ("AUC Score (Train): %f" % roc_auc_score(y_train, y_trainprob))
print ("AUC Score (Test): %f" % roc_auc_score(y_test, y_testprob))

param_test1 = {'colsample_bytree':[0.2,0.4,0.8,0.9,0.95,1.0]} # 列抽样的比率
model=xgb.XGBClassifier(**de_param)
gsearch1 = GridSearchCV(estimator = model,param_grid = param_test1, scoring='roc_auc',cv=3)

gsearch1.fit(X,y)
gsearch1.best_estimator_, gsearch1.best_params_, gsearch1.best_score_

de_param['colsample_bytree']=1

rf0 = xgb.XGBClassifier(**de_param)
rf0.fit(X_train,y_train)
y_trainprob = rf0.predict_proba(X_train)[:,1]
y_testprob = rf0.predict_proba(X_test)[:,1]
print ("AUC Score (Train): %f" % roc_auc_score(y_train, y_trainprob))
print ("AUC Score (Test): %f" % roc_auc_score(y_test, y_testprob))

param_test1 = {'reg_alpha':[0,0.2,0.4,0.8,1.0]}
model=xgb.XGBClassifier(**de_param)
gsearch1 = GridSearchCV(estimator = model,param_grid = param_test1, scoring='roc_auc',cv=3)

gsearch1.fit(X,y)
gsearch1.best_estimator_, gsearch1.best_params_, gsearch1.best_score_

reg_alpha=0

rf0 = xgb.XGBClassifier(**de_param)
rf0.fit(X_train,y_train)
y_trainprob = rf0.predict_proba(X_train)[:,1]
y_testprob = rf0.predict_proba(X_test)[:,1]
print ("AUC Score (Train): %f" % roc_auc_score(y_train, y_trainprob))
print ("AUC Score (Test): %f" % roc_auc_score(y_test, y_testprob))

de_param['reg_alpha']=0

param_test1 = {'reg_alpha':[0,0.01,0.05,0.1]} # 学习率的调节，防止过拟合风险的参数
model=xgb.XGBClassifier(**de_param)
gsearch1 = GridSearchCV(estimator = model,param_grid = param_test1, scoring='roc_auc',cv=3)

gsearch1.fit(X,y)
gsearch1.best_estimator_, gsearch1.best_params_, gsearch1.best_score_

reg_alpha=0.1

rf0 = xgb.XGBClassifier(**de_param)
rf0.fit(X_train,y_train)
y_trainprob = rf0.predict_proba(X_train)[:,1]
y_testprob = rf0.predict_proba(X_test)[:,1]
print ("AUC Score (Train): %f" % roc_auc_score(y_train, y_trainprob))
print ("AUC Score (Test): %f" % roc_auc_score(y_test, y_testprob))

rf0.get_params

de_param