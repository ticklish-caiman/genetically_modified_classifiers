import sys

import pandas as pd
from daal4py.sklearn.decomposition import PCA
from daal4py.sklearn.linear_model import LogisticRegression
from daal4py.sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, Binarizer, PowerTransformer, \
    QuantileTransformer
from sklearn.svm import LinearSVC, NuSVC
from tpot.builtins import StackingEstimator
from xgboost import XGBClassifier

from utils import adjust_dataset

print(sys.maxsize)

if 109387354633633727129018706001594737087405723691181995809612796385783148108984970605 > sys.maxsize:
    print(True)

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

pipe10 = Pipeline(steps=[('nusvc',
                          NuSVC(cache_size=448, coef0=0.11011188298062696, degree=2,
                                nu=0.19806631227688198, random_state=13,
                                tol=9.769099811116575e-08))])

dataset = pd.read_csv('data-CSV/sonar.all-data.csv', delimiter=',')
dataset = adjust_dataset(dataset)
features = dataset.drop('class', axis=1).values

x_train, x_test, y_train, y_test = train_test_split(features, dataset['class'].values, test_size=0.1,
                                                    random_state=13)

for i in range(10):
    cv = cross_val_score(pipe10, x_train, y_train, cv=10, n_jobs=-1,
                         error_score="raise")
    pipe10.fit(x_train, y_train)
    print(f'{cv=}')
    print(f'CV average: {sum(cv) / len(cv)}')
    print(f'Test score: {pipe10.score(x_test, y_test)}')
