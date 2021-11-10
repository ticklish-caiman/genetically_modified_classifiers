import sys
from statistics import stdev

import pandas as pd
from daal4py.sklearn.decomposition import PCA
from daal4py.sklearn.ensemble import RandomForestClassifier
from daal4py.sklearn.linear_model import LogisticRegression
from daal4py.sklearn.neighbors import KNeighborsClassifier
from daal4py.sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, BaggingClassifier, VotingClassifier, \
    StackingClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, Binarizer, PowerTransformer, \
    QuantileTransformer
from sklearn.svm import LinearSVC, NuSVC
from tpot.builtins import StackingEstimator, OneHotEncoder
from xgboost import XGBClassifier

from utils import adjust_dataset

print(sys.maxsize)

tpot_bests_biodeg = []

tpot_bests_biodeg.append(make_pipeline(StackingEstimator(estimator=GaussianNB()),
                                       ExtraTreesClassifier(bootstrap=True, criterion="gini", max_features=0.8,
                                                            min_samples_leaf=1, min_samples_split=6, n_estimators=100)))
tpot_bests_biodeg.append(
    make_pipeline(StackingEstimator(estimator=LogisticRegression(C=15.0, dual=False, penalty="l2")),
                  XGBClassifier(learning_rate=0.1, max_depth=3, min_child_weight=2, n_estimators=100,
                                nthread=1, subsample=0.9500000000000001)))
tpot_bests_biodeg.append(make_pipeline(StackingEstimator(estimator=GaussianNB()), MinMaxScaler(),
                                       ExtraTreesClassifier(bootstrap=True, criterion="entropy",
                                                            max_features=0.9500000000000001, min_samples_leaf=1,
                                                            min_samples_split=10, n_estimators=100)))
tpot_bests_biodeg.append(make_pipeline(StackingEstimator(estimator=GaussianNB()),
                                       RandomForestClassifier(bootstrap=False, criterion="entropy", max_features=0.3,
                                                              min_samples_leaf=1, min_samples_split=6,
                                                              n_estimators=100)))
tpot_bests_biodeg.append(make_pipeline(StackingEstimator(estimator=GaussianNB()),
                                       ExtraTreesClassifier(bootstrap=True, criterion="entropy",
                                                            max_features=0.7000000000000001, min_samples_leaf=3,
                                                            min_samples_split=2, n_estimators=100)))

dataset = pd.read_csv('data-CSV/biodeg.csv', delimiter=',')
dataset = adjust_dataset(dataset)
features = dataset.drop('class', axis=1).values

x_train, x_test, y_train, y_test = train_test_split(features, dataset['class'].values, test_size=0.1,
                                                    random_state=13)

pipes_str = []
results_cv = []
results_test = []
std = []

# for x in tpot_bests_biodeg:
#     cv = cross_val_score(x, x_train, y_train, cv=10, n_jobs=-1,
#                          error_score="raise")
#     x.fit(x_train, y_train)
#     pipes_str.append(str(x))
#     results_cv.append(cv)
#     results_test.append(x.score(x_test, y_test))

tpot_bests_biodeg[0] = Pipeline(steps=[('robustscaler', RobustScaler(quantile_range=(2, 5), with_scaling=False)), (
'pca', PCA(iterated_power=12, n_components=15, random_state=13, tol=0.721712793981146)), ('mlpclassifier',
                                                                                          MLPClassifier(
                                                                                              alpha=0.0002461966497039305,
                                                                                              beta_1=0.8, beta_2=0.8,
                                                                                              epsilon=6.744452314776567e-09,
                                                                                              hidden_layer_sizes=389,
                                                                                              learning_rate='invscaling',
                                                                                              learning_rate_init=5.838171481172468e-05,
                                                                                              max_fun=1066090,
                                                                                              max_iter=4864,
                                                                                              momentum=0.6,
                                                                                              n_iter_no_change=3119,
                                                                                              power_t=0.17848897677742048,
                                                                                              random_state=13,
                                                                                              solver='lbfgs',
                                                                                              tol=8.645569393437859e-09,
                                                                                              validation_fraction=0.05158356193854937))])
for n in range(1):
    cv = cross_val_score(tpot_bests_biodeg[0], x_train, y_train, cv=10, n_jobs=-1,
                         error_score="raise")
    tpot_bests_biodeg[0].fit(x_train, y_train)
    pipes_str.append(str(tpot_bests_biodeg[0]))
    results_cv.append(cv)
    results_test.append(tpot_bests_biodeg[0].score(x_test, y_test))
    std.append(stdev(cv))

cv_averages = []

for i in range(len(pipes_str)):
    cv_averages.append(sum(results_cv[i]) / len(results_cv[i]))
    print(
        f"{pipes_str[i]}\nCV:{results_cv[i]}\nAVG:{sum(results_cv[i]) / len(results_cv[i])}\nTest:{results_test[i]}\nSTDEV:{std[i]}")

print(f"STDEV IN ALL RUNS:{stdev(cv_averages)}")

# from sklearn.utils import all_estimators
#
# estimators = all_estimators()
# print(f"{estimators=}")
# for name, class_ in estimators:
#     if hasattr(class_, 'predict_proba'):
#         print(name)
#
pipe = Pipeline(steps=[('minmaxscaler', MinMaxScaler(feature_range=(3, 4))),
                       ('robustscaler',
                        RobustScaler(quantile_range=(7, 8), with_scaling=False)),
                       ('linearsvc',
                        LinearSVC(C=0.2002209024450622, max_iter=2146000000,  # 5483273010
                                  multi_class='crammer_singer', random_state=13,
                                  tol=0.3372926439555809))])

pipe2 = Pipeline(steps=[('standardscaler',
                         StandardScaler(with_mean=False, with_std=False)),
                        ('robustscaler',
                         RobustScaler(quantile_range=(0, 2), with_scaling=False)),
                        ('pca',
                         PCA(iterated_power=1, n_components=16, random_state=13,
                             svd_solver='randomized', tol=1.474857063305668e+24)),
                        ('binarizer', Binarizer(threshold=0.16751427421303633)),
                        ('powertransformer', PowerTransformer()),
                        ('quantiletransformer',
                         QuantileTransformer(n_quantiles=62, random_state=13,
                                             subsample=672070)),
                        ('svc',
                         SVC(C=3.237811182916519e+32, cache_size=12000,
                             coef0=0.18154572728383522, degree=943522412898845,
                             kernel='linear', max_iter=1, random_state=13,
                             tol=0.13730110260622871))])

pipe3 = Pipeline(steps=[('minmaxscaler', MinMaxScaler(feature_range=(0, 2))),
                        ('robustscaler', RobustScaler(quantile_range=(1, 2))),
                        ('pca',
                         PCA(iterated_power=981247, n_components=18, random_state=13,
                             svd_solver='full', tol=0.16916036222448358, whiten=True)),
                        ('powertransformer', PowerTransformer()),
                        ('quantiletransformer',
                         QuantileTransformer(n_quantiles=46, random_state=13,
                                             subsample=12899214)),
                        ('nusvc',
                         NuSVC(cache_size=8000, coef0=0.11131255758721093,
                               degree=100000000, max_iter=1,
                               nu=0.6068479033102463, random_state=13, shrinking=False,
                               tol=0.10361680952905557))])

pipe4 = Pipeline(steps=[('standardscaler', StandardScaler()),
                        ('minmaxscaler', MinMaxScaler(feature_range=(1, 4))),
                        ('robustscaler', RobustScaler(quantile_range=(0, 2))),
                        ('logisticregression',
                         LogisticRegression(C=7443275351998.929,
                                            l1_ratio=0.38109880448570654,
                                            max_iter=2146000000, random_state=13,
                                            solver='sag', tol=0.23378803458994984))])

pipe5 = Pipeline(
    steps=[('minmaxscaler', MinMaxScaler(feature_range=(1, 7))), ('svc', SVC(C=589.9957663597005, cache_size=222,
                                                                             coef0=0.019974423358650552, degree=2,
                                                                             random_state=13,
                                                                             tol=0.0009100967126519034))])

pipe6 = make_pipeline(StackingEstimator(estimator=GaussianNB()),
                      ExtraTreesClassifier(bootstrap=True, criterion="gini", max_features=0.8, min_samples_leaf=1,
                                           min_samples_split=6, n_estimators=100, random_state=13))

pipe7 = make_pipeline(StackingEstimator(estimator=GaussianNB()), MinMaxScaler(),
                      ExtraTreesClassifier(bootstrap=True, criterion="entropy", max_features=0.9500000000000001,
                                           min_samples_leaf=1, min_samples_split=10, n_estimators=100))

pipe8 = Pipeline(steps=[('standardscaler', StandardScaler()),
                        ('minmaxscaler', MinMaxScaler(feature_range=(2, 4))),
                        ('robustscaler', RobustScaler(quantile_range=(0, 2))),
                        ('svc',
                         SVC(C=23223739.406783454, cache_size=4000,
                             coef0=0.03418055232596423, degree=2, random_state=13,
                             tol=0.00097954302558364))])

pipe9 = Pipeline(steps=[('minmaxscaler', MinMaxScaler(feature_range=(6, 9))),
                        ('robustscaler', RobustScaler(quantile_range=(20, 50))),
                        ('standardscaler', StandardScaler()),
                        ('svc', SVC(C=50.0, kernel='poly', random_state=13))])

pipe10 = Pipeline(steps=[('baggingclassifier',
                          BaggingClassifier(n_estimators=10, random_state=13))])


# StackingClassifier() - can we use it to 'crossover' different types of classifiers?
# ('svc', SVC(C=50.0, kernel='poly', random_state=13)),
pipe11 = Pipeline(steps=[('votingclassifier',
                          VotingClassifier(
                              estimators=[('logisticregression',
                                           LogisticRegression(
                                               C=7443275351998.929,
                                               l1_ratio=0.38109880448570654,
                                               max_iter=2146000000,
                                               random_state=13,
                                               solver='sag',
                                               tol=0.23378803458994984)),
                                          ('baggingclassifier',
                                           BaggingClassifier(n_estimators=10, random_state=13))], voting='soft')),
                         ('nusvc',
                          NuSVC(cache_size=8000, coef0=0.11131255758721093,
                                degree=100000000, max_iter=1,
                                nu=0.6068479033102463, random_state=13, shrinking=False,
                                tol=0.10361680952905557))])

pipe11 = make_pipeline(
    OneHotEncoder(minimum_fraction=0.05, sparse=False, threshold=10),
    MLPClassifier(alpha=0.01, learning_rate_init=0.01)
)

pipe11 = Pipeline(steps=[('minmaxscaler', MinMaxScaler(feature_range=(2, 3))), ('powertransformer', PowerTransformer()),
                         ('kneighborsclassifier',
                          KNeighborsClassifier(algorithm='brute', leaf_size=21, n_neighbors=1, weights='distance'))])
