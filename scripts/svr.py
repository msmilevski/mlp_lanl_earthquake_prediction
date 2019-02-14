import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

x_train = pd.read_csv("../data/x_train.csv")
y_train = pd.read_csv("../data/y_train.csv")
x_val = pd.read_csv("../data/x_val.csv")
y_val = pd.read_csv("../data/y_val.csv")


for C in np.arange(0, 5, 0.1):
    for kernel in ['rbf', 'poly', 'sigmoid', 'linear']:
        for epsilon in [1.0, 2.0, 0.1, 0.01, 0.001]:
            clf = SVR(kernel=kernel, C=C, epsilon=epsilon)
            clf.fit(x_train, y_train)
            predictions = clf.predict(x_val)
            error = mean_absolute_error(y_val, predictions)

            print('Kernel: ' + kernel +', C: ' + str(C) + ', epsilon: ' + str(epsilon) +', MAE: ' + str(error))