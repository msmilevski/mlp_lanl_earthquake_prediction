import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import data_provider
import time

results_path = "SVR_erro_bar.out"

def write_to_results(text, mode="a"):
    with open(results_path, mode) as f:
        f.write(text + "\n")

x_train, y_train, x_val, y_val = data_provider.get_overlapped_data()


def do_shit():
    models_evaluated = 0
    Cs = np.arange(0.01, 5.2, 0.4)
    kernels = ['rbf', 'sigmoid', 'linear']
    epsilons = [1.0, 2.0, 0.1]
    total_num_models = len(kernels) * len(Cs) * len(epsilons)

    def get_error_bar(errors):
        errors = np.array(errors)
        mean = np.mean(errors)
        std = np.std(errors)
        n = len(errors)

        return mean, std / np.sqrt(n)

    models = [
        (0.41, 'linear', 1),
        (0.01, 'linear', 1),
        (0.01, 'linear', 2)
    ]
    seeds = [123131, 856856, 293420]#, 775241, 5562960] 

    print("starting evaluation of 10 models")

    # for C in Cs:
    #     for kernel in kernels:
    #         for epsilon in epsilons:
    for C, kernel, epsilon in models:
        errors = []    
        for seed in seeds:
            print("MODEL {0}".format(models_evaluated))        

            start = time.time()

            clf = SVR(kernel=kernel, C=C, epsilon=epsilon)
            clf.fit(x_train, y_train)
            predictions = clf.predict(x_val)
            error = mean_absolute_error(y_val, predictions)
            errors.append(error)

            
            
            # write_to_results(result)
            result = 'Kernel: ' + kernel +', C: ' + str(C) + ', epsilon: ' + str(epsilon) +', MAE: ' + str(error)
            print(result)

            end = time.time()
            print("Model {0} evaluation finished, time elapsed: {1}".format(models_evaluated, end - start))
            models_evaluated +=1

            print("{0}/{1} models evaluated".format(models_evaluated, total_num_models))

        mean, error_bar = get_error_bar(errors)
        model = 'Kernel: ' + kernel +', C: ' + str(C) + ', epsilon: ' + str(epsilon)
        write_to_results("{0}, error: {1} +- {2}".format(model, mean, error_bar))

def evaluate_test():

    x_test = data_provider.get_test_x()

    clf = SVR(kernel='linear', C=0.41, epsilon=1)

    print('fitting')
    clf.fit(x_train, y_train)

    


    test_predictions = clf.predict(x_test)

    submission = pd.read_csv('data/sample_submission.csv')
    submission['time_to_failure'] = test_predictions
    submission.to_csv("submission.csv", index=False)

evaluate_test()

