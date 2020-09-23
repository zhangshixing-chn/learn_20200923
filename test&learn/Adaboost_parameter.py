# base_estimator:基分类器，默认是决策树，在该分类器基础上进行boosting，理论上可以是任意一个分类器，
# 但是如果是其他分类器时需要指明样本权重。

# n_estimators:基分类器提升（循环）次数，默认是50次，这个值过大，模型容易过拟合；值过小，模型容易欠拟合。

# learning_rate:学习率，表示梯度收敛速度，默认为1，如果过大，容易错过最优值，如果过小，则收敛速度会很慢；
# 该值需要和n_estimators进行一个权衡，当分类器迭代次数较少时，学习率可以小一些，当迭代次数较多时，学习率可以适当放大。

# algorithm:boosting算法，也就是模型提升准则，有两种方式SAMME, 和SAMME.R两种，默认是SAMME.R，
# 两者的区别主要是弱学习器权重的度量，前者是对样本集预测错误的概率进行划分的，后者是对样本集的预测错误的比例，
# 即错分率进行划分的，默认是用的SAMME.R。

import numpy as np
import pandas as pd
np.random.seed(10)
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

X, y = make_classification(n_samples=800)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

rf0 = AdaBoostClassifier(random_state=10)
rf0.fit(X_train,y_train)
y_trainprob = rf0.predict_proba(X_train)[:,1]
y_testprob = rf0.predict_proba(X_test)[:,1]
print ("AUC Score (Train): %f" % roc_auc_score(y_train, y_trainprob))
print ("AUC Score (Test): %f" % roc_auc_score(y_test, y_testprob))

rf0.get_params

param_test1 = {'n_estimators':[1,3,5,7,10,50,90,130,170],'learning_rate':[0.2,0.4,0.6,0.8]}
gsearch1 = GridSearchCV(estimator = AdaBoostClassifier(random_state=10),
                                    param_grid = param_test1, scoring='roc_auc',cv=3)

gsearch1.fit(X_train,y_train)

gsearch1.best_estimator_,gsearch1.best_params_,gsearch1.best_score_


rf0 = AdaBoostClassifier(n_estimators=10,learning_rate=0.8)
rf0.fit(X_train,y_train)

y_trainprob = rf0.predict_proba(X_train)[:,1]
y_testprob = rf0.predict_proba(X_test)[:,1]
print ("AUC Score (Train): %f" % roc_auc_score(y_train, y_trainprob))
print ("AUC Score (Test): %f" % roc_auc_score(y_test, y_testprob))