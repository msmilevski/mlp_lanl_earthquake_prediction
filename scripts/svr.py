import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import data_provider
import time

results_path = "SVR.out"

def write_to_results(text, mode="a"):
    with open(results_path, mode) as f:
        f.write(text + "\n")

x_train, y_train, x_val, y_val = data_provider.get_data()

models_evaluated = 0
Cs = np.arange(0.01, 5.2, 0.4)
kernels = ['rbf', 'sigmoid', 'linear']
epsilons = [1.0, 2.0, 0.1]
total_num_models = len(kernels) * len(Cs) * len(epsilons)

print("starting evaluation of {0} models".format(total_num_models))
for C in Cs:
    for kernel in kernels:
        for epsilon in epsilons:
            print("MODEL {0}".format(models_evaluated))

            if models_evaluated <= 7:   # Already have results for these models
                print("skipping this model")
                models_evaluated += 1
                continue

            
            start = time.time()

            clf = SVR(kernel=kernel, C=C, epsilon=epsilon)
            clf.fit(x_train, y_train)
            predictions = clf.predict(x_val)
            error = mean_absolute_error(y_val, predictions)

            result = 'Kernel: ' + kernel +', C: ' + str(C) + ', epsilon: ' + str(epsilon) +', MAE: ' + str(error)
            print(result)
            write_to_results(result)

            end = time.time()
            print("Model {0} evaluation finished, time elapsed: {1}".format(models_evaluated, end - start))
            models_evaluated +=1

            print("{0}/{1} models evaluated".format(models_evaluated, total_num_models))

