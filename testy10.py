import sys

import pandas as pd
from daal4py.sklearn.decomposition import PCA
from daal4py.sklearn.linear_model import LogisticRegression
from daal4py.sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, Binarizer, PowerTransformer, \
    QuantileTransformer
from sklearn.svm import LinearSVC, NuSVC

print(sys.maxsize)

# No więc dodamy zabezpieczenie
# ale dlaczego to się nie wczytuje:
# Pipeline(steps=[('minmaxscaler', MinMaxScaler(feature_range=(1, 7))), ('svc', SVC(C=589.9957663597005, cache_size=222,
# coef0=0.019974423358650552, degree=2, random_state=13, tol=0.0009100967126519034))])
if 109387354633633727129018706001594737087405723691181995809612796385783148108984970605 > sys.maxsize:
    print(True)

pipe = Pipeline(steps=[('minmaxscaler', MinMaxScaler(feature_range=(3, 4))),
                ('robustscaler',
                 RobustScaler(quantile_range=(7, 8), with_scaling=False)),
                ('linearsvc',
                 LinearSVC(C=0.2002209024450622, max_iter=2146000000, #5483273010
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

dataset = pd.read_csv('data-CSV/biodeg.csv', delimiter=',')
features = dataset.drop('class', axis=1).values
x_train, x_test, y_train, y_test = train_test_split(features, dataset['class'].values, test_size=0.1,
                                                    random_state=13)
pipe4.fit(x_train, y_train)
print(pipe4.score(x_test, y_test))
