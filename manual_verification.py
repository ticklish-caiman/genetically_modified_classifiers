from datetime import datetime
import pickle
import sys
from collections import Counter
from statistics import stdev

import pandas as pd
from daal4py.sklearn.decomposition import PCA
from daal4py.sklearn.ensemble import RandomForestClassifier
from daal4py.sklearn.linear_model import LogisticRegression
from daal4py.sklearn.neighbors import KNeighborsClassifier
from daal4py.sklearn.svm import SVC
from sklearn.decomposition import FastICA
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, BaggingClassifier, VotingClassifier, \
    StackingClassifier
from sklearn.feature_selection import SelectPercentile
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, Binarizer, PowerTransformer, \
    QuantileTransformer, FunctionTransformer
from sklearn.svm import LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator, OneHotEncoder, CombineDFs
from tpot.export_utils import set_param_recursive
from xgboost import XGBClassifier
from copy import copy

from utils import adjust_dataset
from scipy.stats import friedmanchisquare, ttest_rel, wilcoxon

tpot_best_sonar01 = Pipeline(steps=[('stackingestimator', StackingEstimator(
    estimator=GradientBoostingClassifier(learning_rate=0.001, max_depth=4, max_features=0.6000000000000001,
                                         min_samples_leaf=12, min_samples_split=3, subsample=0.1))),
                                    ('minmaxscaler', MinMaxScaler()), ('extratreesclassifier',
                                                                       ExtraTreesClassifier(criterion='entropy',
                                                                                            max_features=0.55,
                                                                                            min_samples_split=3))])

Pipeline(steps=[('selectpercentile', SelectPercentile(percentile=95)), ('standardscaler', StandardScaler()), (
    'stackingestimator', StackingEstimator(
        estimator=SGDClassifier(alpha=0.0, eta0=0.1, fit_intercept=False, l1_ratio=0.0, learning_rate='invscaling',
                                penalty='elasticnet', power_t=50.0))),
                ('mlpclassifier', MLPClassifier(alpha=0.001))])

Pipeline(steps=[('standardscaler', StandardScaler()),
                ('robustscaler', RobustScaler(quantile_range=(3, 4), with_centering=False)),
                ('powertransformer', PowerTransformer()),
                ('gaussianprocessclassifier',
                 GaussianProcessClassifier(max_iter_predict=1628, random_state=None))])

gmc_best_sonar02 = Pipeline(
    steps=[('standardscaler', StandardScaler(with_std=False)), ('minmaxscaler', MinMaxScaler(feature_range=(4, 5))),
           ('powertransformer', PowerTransformer()), ('nusvc',
                                                      NuSVC(cache_size=523, coef0=0.06919019288956918, degree=4,
                                                            nu=0.3590416064855505, tol=0.0007025770443640796,
                                                            random_state=None))])

gmc_best_sonar03 = Pipeline(steps=[('standardscaler', StandardScaler(with_mean=False, with_std=False)),
                                   ('minmaxscaler', MinMaxScaler(feature_range=(4, 9))),
                                   ('robustscaler', RobustScaler(quantile_range=(0, 1), with_centering=False)),
                                   ('powertransformer', PowerTransformer()), ('gaussianprocessclassifier',
                                                                              GaussianProcessClassifier(
                                                                                  max_iter_predict=304,
                                                                                  n_restarts_optimizer=35,
                                                                                  optimizer=0.9304558217664426,
                                                                                  random_state=None))])
gmc_best_sonar04 = Pipeline(
    steps=[('minmaxscaler', MinMaxScaler(feature_range=(6, 7))), ('powertransformer', PowerTransformer()),
           ('quantiletransformer', QuantileTransformer(n_quantiles=14, subsample=827228)), ('kneighborsclassifier',
                                                                                            KNeighborsClassifier(
                                                                                                algorithm='ball_tree',
                                                                                                leaf_size=371,
                                                                                                n_neighbors=1,
                                                                                                weights='distance'))])

gmc_sonar_svc = Pipeline(steps=[('minmaxscaler', MinMaxScaler(feature_range=(2, 4))),
                                ('robustscaler', RobustScaler(quantile_range=(71, 76))), ('svc',
                                                                                          SVC(C=109.79683786260674,
                                                                                              cache_size=297,
                                                                                              coef0=0.027350032779296502,
                                                                                              degree=2,
                                                                                              tol=0.0010080380598635605))])

Pipeline(steps=[('pca', PCA(iterated_power=8, n_components=14, tol=2.669680756488349)), (
    'mlpclassifier', MLPClassifier(alpha=0.00010119183095890908, epsilon=9.8591699830632e-09, hidden_layer_sizes=72,
                                   learning_rate='adaptive', learning_rate_init=9.742972195186764e-05, max_fun=38941,
                                   max_iter=2217, momentum=0.6, n_iter_no_change=124, power_t=0.4688215167466403,
                                   solver='lbfgs', tol=9.660414969538923e-09,
                                   validation_fraction=0.10537113889958741))])

# TPOT MAGIC 100% 32/32 Random:13
magic_tpot = make_pipeline(
    make_union(
        FastICA(tol=0.45),
        FunctionTransformer(copy)
    ),
    XGBClassifier(learning_rate=0.1, max_depth=9, min_child_weight=6, n_estimators=100, n_jobs=1,
                  subsample=0.9000000000000001, verbosity=0)
)

magic_gmc_best = Pipeline(steps=[('standardscaler', StandardScaler(with_mean=False)), (
    'svc', SVC(C=410.4657558903129, cache_size=268, coef0=0.10497752576001129, tol=0.001099514046179941))])

dataset = pd.read_csv('data-CSV/sonar.all-data.csv', delimiter=',')
dataset = adjust_dataset(dataset)
features = dataset.drop('class', axis=1).values

cv_full_results = []
cv_results = []
stds = []
test_results = []
times = []
header = "CV,TEST,STDV(CV),TIME,FULL_CV\n"

pipe_to_test = tpot_best_sonar01

for step in pipe_to_test.steps:
    print(step[1])

StackingEstimator(estimator=GradientBoostingClassifier(learning_rate=0.001,
                                                       max_depth=4,
                                                       max_features=0.6000000000000001,
                                                       min_samples_leaf=12,
                                                       min_samples_split=3,
                                                       subsample=0.1))
MinMaxScaler()
ExtraTreesClassifier(criterion='entropy', max_features=0.55,
                     min_samples_split=3)

x_train, x_test, y_train, y_test = train_test_split(features, dataset['class'].values, test_size=0.1,
                                                    random_state=None)
print(pipe_to_test.score(x_test, y_test, ))

# set_param_recursive(pipe_to_test.steps, 'random_state', None)
# if hasattr(pipe_to_test, 'random_state'):
#     setattr(pipe_to_test, 'random_state', None)
#
# # ------------------------------------------------           100x dla różnych splitów
# for i in range(100):
#     start_time = datetime.now()
#     x_train, x_test, y_train, y_test = train_test_split(features, dataset['class'].values, test_size=0.1,
#                                                         random_state=None)
#     cv = cross_val_score(pipe_to_test, x_train, y_train, cv=10, n_jobs=-1,
#                          error_score="raise")
#     times.append(datetime.now() - start_time)
#     print(i + 1)
#     cv_full_results.append(str(cv))
#     cv_results.append(sum(cv) / len(cv))
#     stds.append(stdev(cv))
#     pipe_to_test.fit(x_train, y_train)
#     test_results.append(pipe_to_test.score(x_test, y_test))
#
# line = ""
# for i in range(100):
#     cv_full_results[i] = [
#         cv_full_results[i].replace("\r\n", "").replace("  ", "").replace("\n", "").replace("[", "").replace("]",
#                                                                                                             "").replace(
#             "\'", "") for x
#         in
#         cv_full_results[i]]
#     line += f"{cv_results[i]},{test_results[i]},{stds[i]},{times[i]},{cv_full_results[i][0]}\n"
# with open('different_splits.csv', 'w') as handle:
#     print(f"{header}{line}", file=handle)
#
# print("Half way there")
# cv_full_results = []
# cv_results = []
# stds = []
# test_results = []
# times = []
#
# # -----------------------------------------------------------            100x dla tego samego splitu
# x_train, x_test, y_train, y_test = train_test_split(features, dataset['class'].values, test_size=0.1,
#                                                     random_state=13)
# for i in range(100):
#     start_time = datetime.now()
#     cv = cross_val_score(pipe_to_test, x_train, y_train, cv=10, n_jobs=-1,
#                          error_score="raise")
#     times.append(datetime.now() - start_time)
#     print(i + 1)
#     cv_full_results.append(str(cv))
#     cv_results.append(sum(cv) / len(cv))
#     stds.append(stdev(cv))
#     pipe_to_test.fit(x_train, y_train)
#     test_results.append(pipe_to_test.score(x_test, y_test))
#
# line = ""
# for i in range(100):
#     cv_full_results[i] = [
#         cv_full_results[i].replace("\r\n", "").replace("  ", "").replace("\n", "").replace("[", "").replace("]",
#                                                                                                             "").replace(
#             "\'", "") for x
#         in
#         cv_full_results[i]]
#     line += f"{cv_results[i]},{test_results[i]},{stds[i]},{times[i]},{cv_full_results[i][0]}\n"
#
# with open('same_splits.csv', 'w') as handle:
#     print(f"{header}{line}", file=handle)
#
# print('Finito')

# # all operator 216
# Pipeline(steps=[('minmaxscaler', MinMaxScaler(feature_range=(0, 7))),
#                 ('robustscaler', RobustScaler(quantile_range=(3, 6), with_centering=False)),
#                 ('powertransformer', PowerTransformer()), ('gaussianprocessclassifier',
#                                                            GaussianProcessClassifier(max_iter_predict=122,
#                                                                                      optimizer=0.09911181319287213))])
# # mutation only 216
# Pipeline(steps=[('standardscaler', StandardScaler(with_mean=False, with_std=False)), ('minmaxscaler', MinMaxScaler()),
#                 ('robustscaler', RobustScaler(quantile_range=(25, 27), with_centering=False)),
#                 ('powertransformer', PowerTransformer()), ('gaussianprocessclassifier',
#                                                            GaussianProcessClassifier(max_iter_predict=148,
#                                                                                      optimizer=0.09370535990502449))])
# 10%ofMAGIC 0.8573847409220727 Time: 0:57:19.945518
# nie testować na pełnym zbiorze! Chyba, że ma się 1TB RAM
# monster = Pipeline(steps=[('standardscaler', StandardScaler(with_std=False)),
#                           ('robustscaler',
#                            RobustScaler(quantile_range=(41, 48), with_centering=False)),
#                           ('powertransformer', PowerTransformer()),
#                           ('gaussianprocessclassifier',
#                            GaussianProcessClassifier(max_iter_predict=239,
#                                                      multi_class='one_vs_one',
#                                                      n_restarts_optimizer=13,
#                                                      optimizer=0.09534910665786302))])
