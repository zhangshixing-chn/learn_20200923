# 集成类参数
# 1) n_estimators: 也就是弱学习器的最大迭代次数，或者说最大的弱学习器的个数。
#     一般来说n_estimators太小，容易欠拟合，n_estimators太大，计算量会太大，
#     并且n_estimators到一定的数量后，再增大n_estimators获得的模型提升会很小，
#     所以一般选择一个适中的数值。默认是100。

# 有关树结构的参数
# 1) RF划分时考虑的最大特征数max_features: 可以使用很多种类型的值，默认是"auto",意味着划分时最多考虑N个特征；
#     如果是"log2"意味着划分时最多考虑log2N个特征；
#     如果是"sqrt"或者"auto"意味着划分时最多考虑sqrt（N）个特征。如果是整数，代表考虑的特征绝对数。
#     如果是浮点数，代表考虑特征百分比，即考虑（百分比xN）取整后的特征数。其中N为样本总特征数。
#     一般我们用默认的"auto"就可以了，如果特征数非常多，我们可以灵活使用刚才描述的其他取值来控制划分时考虑的最大特征数，以控制决策树的生成时间。

# 2) 决策树最大深度max_depth: 默认可以不输入，如果不输入的话，决策树在建立子树的时候不会限制子树的深度。
#     一般来说，数据少或者特征少的时候可以不管这个值。
#     如果模型样本量多，特征也多的情况下，推荐限制这个最大深度，具体的取值取决于数据的分布。常用的可以取值10-100之间。

# 3) 内部节点再划分所需最小样本数min_samples_split: 这个值限制了子树继续划分的条件，
#     如果某节点的样本数少于min_samples_split，则不会继续再尝试选择最优特征来进行划分。 
#     默认是2.如果样本量不大，不需要管这个值。如果样本量数量级非常大，则推荐增大这个值。

# 4) 叶子节点最少样本数min_samples_leaf: 这个值限制了叶子节点最少的样本数，如果某叶子节点数目小于样本数，则会和兄弟节点一起被剪枝。 
#     默认是1,可以输入最少的样本数的整数，或者最少样本数占样本总数的百分比。如果样本量不大，不需要管这个值。
#     如果样本量数量级非常大，则推荐增大这个值。

# 5）叶子节点最小的样本权重和min_weight_fraction_leaf：这个值限制了叶子节点所有样本权重和的最小值，
# 如果小于这个值，则会和兄弟节点一起被剪枝。 默认是0，就是不考虑权重问题。
# 一般来说，如果我们有较多样本有缺失值，或者分类树样本的分布类别偏差很大，就会引入样本权重，这时我们就要注意这个值了。

# 6) 最大叶子节点数max_leaf_nodes: 通过限制最大叶子节点数，可以防止过拟合，默认是"None”，即不限制最大的叶子节点数。
#     如果加了限制，算法会建立在最大叶子节点数内最优的决策树。
#     如果特征不多，可以不考虑这个值，但是如果特征分成多的话，可以加以限制，具体的值可以通过交叉验证得到。

# 7) 节点划分最小不纯度min_impurity_split:  这个值限制了决策树的增长，
#     如果某节点的不纯度(基于基尼系数，均方差)小于这个阈值，则该节点不再生成子节点。
#     即为叶子节点 。一般不推荐改动默认值1e-7。

import numpy as np
import pandas as pd
np.random.seed(10)
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

X, y = make_classification(n_samples=800)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

rf0 = GradientBoostingClassifier(random_state=10)
rf0.fit(X_train,y_train)
y_trainprob = rf0.predict_proba(X_train)[:,1]
y_testprob = rf0.predict_proba(X_test)[:,1]
print ("AUC Score (Train): %f" % roc_auc_score(y_train, y_trainprob))
print ("AUC Score (Test): %f" % roc_auc_score(y_test, y_testprob))

rf0.get_params

#get_ipython().run_line_magic('pinfo', 'rf0')

n_estimators=100
max_depth=3
min_samples_leaf=1
min_samples_split=2
max_features=None
subsample=1

param_test1 = {'n_estimators':range(10,600,40)}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(min_samples_split=min_samples_split,subsample=subsample,
                                  min_samples_leaf=min_samples_leaf,max_depth=max_depth,max_features=max_features,random_state=10),
                                    param_grid = param_test1, scoring='roc_auc',cv=3)

# gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(min_samples_split=min_samples_split,subsample=subsample,
#                                   min_samples_leaf=min_samples_leaf,max_depth=max_depth,n_estimators=n_estimators,
#                                   max_features=max_features,random_state=10),
#                                     param_grid = param_test1, scoring='roc_auc',cv=3)

gsearch1.fit(X,y)
gsearch1.get_params, gsearch1.best_params_, gsearch1.best_score_

n_estimators=290


rf0 = GradientBoostingClassifier(random_state=10,min_samples_split=min_samples_split,subsample=subsample,
                                  min_samples_leaf=min_samples_leaf,max_depth=max_depth,
                                 max_features=max_features,n_estimators=n_estimators)
rf0.fit(X_train,y_train)
y_trainprob = rf0.predict_proba(X_train)[:,1]
y_testprob = rf0.predict_proba(X_test)[:,1]
print ("AUC Score (Train): %f" % roc_auc_score(y_train, y_trainprob))
print ("AUC Score (Test): %f" % roc_auc_score(y_test, y_testprob))


param_test1 = {'min_samples_leaf':range(1,8,1),'min_samples_split':range(2,8,2)}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(subsample=subsample,n_estimators=n_estimators,
                                  max_depth=max_depth,max_features=max_features,random_state=10),
                                    param_grid = param_test1, scoring='roc_auc',cv=3)

gsearch1.fit(X,y)
gsearch1.param_grid, gsearch1.best_params_, gsearch1.best_score_

min_samples_leaf=7
min_samples_split=2

rf0 = GradientBoostingClassifier(random_state=10,min_samples_split=min_samples_split,subsample=subsample,
                                  min_samples_leaf=min_samples_leaf,max_depth=max_depth,
                                 max_features=max_features,n_estimators=n_estimators)
rf0.fit(X_train,y_train)
y_trainprob = rf0.predict_proba(X_train)[:,1]
y_testprob = rf0.predict_proba(X_test)[:,1]
print ("AUC Score (Train): %f" % roc_auc_score(y_train, y_trainprob))
print ("AUC Score (Test): %f" % roc_auc_score(y_test, y_testprob))

param_test1 = {'max_features':range(3,20,2)}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(min_samples_split=min_samples_split,subsample=subsample,
                                  min_samples_leaf=min_samples_leaf,max_depth=max_depth,n_estimators=n_estimators,
                                  random_state=10),
                                    param_grid = param_test1, scoring='roc_auc',cv=3)

gsearch1.fit(X,y)

gsearch1.best_estimator_, gsearch1.best_params_, gsearch1.best_score_

max_features=19

rf0 = GradientBoostingClassifier(random_state=10,min_samples_split=min_samples_split,subsample=subsample,
                                  min_samples_leaf=min_samples_leaf,max_depth=max_depth,
                                 max_features=max_features,n_estimators=n_estimators)
rf0.fit(X_train,y_train)
y_trainprob = rf0.predict_proba(X_train)[:,1]
y_testprob = rf0.predict_proba(X_test)[:,1]
print ("AUC Score (Train): %f" % roc_auc_score(y_train, y_trainprob))
print ("AUC Score (Test): %f" % roc_auc_score(y_test, y_testprob))

param_test1 = {'subsample':[0.4,0.5,0.6,0.7,0.8,0.9,1.0]}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(min_samples_split=min_samples_split,
                                  min_samples_leaf=min_samples_leaf,max_depth=max_depth,n_estimators=n_estimators,
                                  max_features=max_features,random_state=10),
                                    param_grid = param_test1, scoring='roc_auc',cv=3)

gsearch1.fit(X,y)

gsearch1.best_estimator_, gsearch1.best_params_, gsearch1.best_score_

subsample=1

rf0 = GradientBoostingClassifier(random_state=10,min_samples_split=min_samples_split,subsample=subsample,
                                  min_samples_leaf=min_samples_leaf,max_depth=max_depth,
                                 max_features=max_features,n_estimators=n_estimators)
rf0.fit(X_train,y_train)
y_trainprob = rf0.predict_proba(X_train)[:,1]
y_testprob = rf0.predict_proba(X_test)[:,1]
print ("AUC Score (Train): %f" % roc_auc_score(y_train, y_trainprob))
print ("AUC Score (Test): %f" % roc_auc_score(y_test, y_testprob))


rf0.get_params

rf1=gsearch1.best_estimator_

rf1.fit(X_train,y_train)

y_trainprob = rf1.predict_proba(X_train)[:,1]
y_testprob = rf1.predict_proba(X_test)[:,1]
print ("AUC Score (Train): %f" % roc_auc_score(y_train, y_trainprob))
print ("AUC Score (Test): %f" % roc_auc_score(y_test, y_testprob))