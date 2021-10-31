import sys

print(sys.maxsize)

# No więc dodamy zabezpieczenie
# ale dlaczego to się nie wczytuje:
# Pipeline(steps=[('minmaxscaler', MinMaxScaler(feature_range=(1, 7))), ('svc', SVC(C=589.9957663597005, cache_size=222,
# coef0=0.019974423358650552, degree=2, random_state=13, tol=0.0009100967126519034))])
