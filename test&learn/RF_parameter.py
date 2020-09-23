#
# 集成类参数
# 1) n_estimators: 也就是弱学习器的最大迭代次数，或者说最大的弱学习器的个数。
#     一般来说n_estimators太小，容易欠拟合，n_estimators太大，计算量会太大，
#     并且n_estimators到一定的数量后，再增大n_estimators获得的模型提升会很小，
#     所以一般选择一个适中的数值。默认是100。

# 2) oob_score :即是否采用袋外样本来评估模型的好坏。默认识False。
# 个人推荐设置为True，因为袋外分数反应了一个模型拟合后的泛化能力。

# 3) criterion: 即CART树做划分时对特征的评价标准。分类模型和回归模型的损失函数是不一样的。
# 分类RF对应的CART分类树默认是基尼系数gini,另一个可选择的标准是信息增益。
# 回归RF对应的CART回归树默认是均方差mse，另一个可以选择的标准是绝对值差mae。一般来说选择默认的标准就已经很好的。

# 有关树结构的参数
# 1) RF划分时考虑的最大特征数max_features: 可以使用很多种类型的值，默认是"auto",意味着划分时最多考虑N个特征；
#     如果是"log2"意味着划分时最多考虑log2(N)个特征；
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

# 引入相关的模块
import numpy as np
import pandas as pd
np.random.seed(10)
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

# 构造数据集
X, y = make_classification(n_samples=800) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

X.shape,y.shape

# ### 采用默认参数

rf0 = RandomForestClassifier(random_state=10)
rf0.fit(X_train,y_train)
y_trainprob = rf0.predict_proba(X_train)[:,1]
y_testprob = rf0.predict_proba(X_test)[:,1]
print ("AUC Score (Train): %f" % roc_auc_score(y_train, y_trainprob))
print ("AUC Score (Test): %f" % roc_auc_score(y_test, y_testprob))

rf0.get_params

n_estimators=10
max_depth=None
min_samples_split=2
min_samples_leaf=1
max_features='auto'

param_test1 = {'n_estimators':range(10,101,10)}
gsearch1 = GridSearchCV(estimator = RandomForestClassifier(min_samples_split=min_samples_split,
                                  min_samples_leaf=min_samples_leaf,max_depth=max_depth,
                                    max_features=max_features,random_state=10),
                                    param_grid = param_test1, scoring='roc_auc',cv=3)

gsearch1.fit(X,y)
gsearch1.best_estimator_, gsearch1.best_params_, gsearch1.best_score_

gsearch1.b


n_estimators=70

# # n_estimators 查看

rf0 = RandomForestClassifier(n_estimators=n_estimators,random_state=10)
rf0.fit(X_train,y_train)
y_trainprob = rf0.predict_proba(X_train)[:,1]
y_testprob = rf0.predict_proba(X_test)[:,1]
print ("AUC Score (Train): %f" % roc_auc_score(y_train, y_trainprob))
print ("AUC Score (Test): %f" % roc_auc_score(y_test, y_testprob))

# # max_depth , min_samples_split 查看

param_test1 = {'max_depth':range(10,41,10),'min_samples_split':range(2,11,2),'min_samples_leaf':range(1,5,1)}
gsearch1 = GridSearchCV(estimator = RandomForestClassifier(
                                  n_estimators=n_estimators,max_features=max_features,random_state=10),
                                    param_grid = param_test1, scoring='roc_auc',cv=3)

gsearch1.fit(X,y)

gsearch1.get_params, gsearch1.best_params_, gsearch1.best_score_

max_depth = 20
min_samples_split = 6
min_samples_leaf = 1

rf0 = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,
                             min_samples_split=min_samples_split,random_state=10)
rf0.fit(X_train,y_train)
y_trainprob = rf0.predict_proba(X_train)[:,1]
y_testprob = rf0.predict_proba(X_test)[:,1]
print ("AUC Score (Train): %f" % roc_auc_score(y_train, y_trainprob))
print ("AUC Score (Test): %f" % roc_auc_score(y_test, y_testprob))

# # min_samples_leaf 

param_test1 = {'min_samples_leaf':range(1,5,1)}
gsearch1 = GridSearchCV(estimator = RandomForestClassifier(max_depth=max_depth,min_samples_split=min_samples_split,
                                  n_estimators=n_estimators,max_features=max_features,random_state=10),
                                    param_grid = param_test1, scoring='roc_auc',cv=3)

gsearch1.fit(X,y)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

min_samples_leaf=1

rf0 = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,random_state=10)
rf0.fit(X_train,y_train)
y_trainprob = rf0.predict_proba(X_train)[:,1]
y_testprob = rf0.predict_proba(X_test)[:,1]
print ("AUC Score (Train): %f" % roc_auc_score(y_train, y_trainprob))
print ("AUC Score (Test): %f" % roc_auc_score(y_test, y_testprob))

# # max_features
param_test1 = {'max_features':[3,4,5,6]}
gsearch1 = GridSearchCV(estimator = RandomForestClassifier(max_depth=max_depth,min_samples_split=min_samples_split,
                                  n_estimators=n_estimators,min_samples_leaf=min_samples_leaf,random_state=10),
                                    param_grid = param_test1, scoring='roc_auc',cv=3)
gsearch1.fit(X,y)

gsearch1.best_estimator_, gsearch1.best_params_, gsearch1.best_score_

max_features=4
rf0 = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,min_samples_split=min_samples_split,
                             min_samples_leaf=min_samples_leaf,max_features=max_features,random_state=10)
rf0.fit(X_train,y_train)
y_trainprob = rf0.predict_proba(X_train)[:,1]
y_testprob = rf0.predict_proba(X_test)[:,1]
print ("AUC Score (Train): %f" % roc_auc_score(y_train, y_trainprob))
print ("AUC Score (Test): %f" % roc_auc_score(y_test, y_testprob))

print("n_estimators:",n_estimators)
print("max_depth:",max_depth)
print("min_samples_split:",min_samples_split)
print("min_samples_leaf:",min_samples_leaf)
print("max_features:",max_features)

rf0.get_params

rf1 = RandomForestClassifier(n_estimators=n_estimators,max_depth=None,
                             min_samples_split=2,min_samples_leaf=1,
                             max_features=4,random_state=10)

rf1.fit(X_train,y_train)

y_trainprob = rf1.predict_proba(X_train)[:,1]
y_testprob = rf1.predict_proba(X_test)[:,1]
print ("AUC Score (Train): %f" % roc_auc_score(y_train, y_trainprob))
print ("AUC Score (Test): %f" % roc_auc_score(y_test, y_testprob))

n_estimators: 70 #越大越容易过拟合
max_depth: 20 #越大越容易过拟合
min_samples_split: 6 #越大越容易欠拟合
min_samples_leaf: 1 #越大越容易欠拟合
max_features: 4 #越大越容易过拟合