import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
import os
import gc
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from scipy.stats import skew
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error
pd.options.display.precision = 15
import data_provider

os.environ['KMP_DUPLICATE_LIB_OK']='True'

x_train, y_train, x_val, y_val = data_provider.get_data()
x_train = x_train.drop(['var_norm_FT10', 'var_norm_FT90'], 1)
x_val = x_val.drop(['var_norm_FT10', 'var_norm_FT90'], 1)


def model_name(max_depth, eta, objective, booster, gamma, reg_alpha, reg_lambda, subsample, nthread):
    return "max_depth: {0}, eta: {1}, objective: {2}, booster:{3}, gamma:{4}, reg_alpha:{5},reg_lambda:{6},subsample:{7},nthread:{8}".format(max_depth, eta, 
                            objective, booster, gamma, reg_alpha, reg_lambda, subsample, nthread)
                            
results_path = os.path.join("XGBoost.csv")
rnd_seed = 123131

def write_to_results(text, mode="a"):
    with open(results_path, mode) as f:
        f.write(text + "\n")

def save_result(error, max_depth, eta, objective, booster, gamma, reg_alpha, reg_lambda, subsample, nthread):
    write_to_results("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}".format(max_depth, eta, objective, booster, gamma, 
                                                  reg_alpha, reg_lambda, subsample, nthread, error))
                                                  
write_to_results("max_depth, eta, objective, booster, gamma, reg_alpha, reg_lambda, subsample, nthread", mode="w") 
 
train_data = xgb.DMatrix(data=x_train, label=y_train, feature_names=x_train.columns)
valid_data = xgb.DMatrix(data=x_val, label=y_val, feature_names=x_train.columns)

watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]

max_depths = [6, 7, 8]#[6, 7, 8, 9, 10]
etas = [0.001, 0.01, 0.05]#[0.001, 0.01, 0.05, 0.1, 0.2, 0.3]
objectives = ['reg:gamma', 'reg:linear']#['reg:linear', 'reg:gamma']
boosters = ['gbtree']
gammas = [0]#[0, 0.1, 0.5]
reg_alphas = [0]#[0, 0.05, 0.1, 0.2, 0.3]
reg_lambdas = [0.2, 0.3]#[0, 0.05, 0.1, 0.2, 0.3]
subsamples = [0.5]#[0.5, 0.6, 0.7, 0.8 , 0.9, 1]
nthreads = [4]

grid_size = len(max_depths) * len(etas) * len(objectives) * \
                len(boosters) * len(gammas) * len(reg_alphas) * len(subsamples) * len(reg_lambdas) * len(nthreads)
                
models_evaluated = 0
for max_depth in max_depths:
    for eta in etas:
        for objective in objectives:
            for booster in boosters:
                for gamma in gammas:
                    for reg_alpha in reg_alphas:
                        for reg_lambda in reg_lambdas:
                            for subsample in subsamples:
                                for nthread in nthreads:
                                    current_model_name = model_name(max_depth, eta, objective, booster, gamma, reg_alpha, reg_lambda, subsample, nthread)
                                    print("Training model:")
                                    print(current_model_name)
                                    
                                    parameters = {'max_depth':max_depth, 'eta':eta, 'objective':objective, 
                                                  'booster':booster, 'gamma':gamma, 'reg_alpha':reg_alpha,
                                                  'reg_lambda':reg_lambda, 'subsample':subsample, 'nthread':nthread, 'silent':True}
                                    
                                    XGBModel = xgb.train(dtrain=train_data, num_boost_round=1000, params=parameters)
                                    y_pred = XGBModel.predict(xgb.DMatrix(x_val, feature_names=x_val.columns))
                                    XGBModel.save_model('model_{0}'.format(models_evaluated + 1))
                                    error = mean_absolute_error(y_val, y_pred)
                                    print(error)

                                    save_result(error, max_depth, eta, objective, booster, gamma, reg_alpha, reg_lambda, subsample, nthread)

                                    models_evaluated += 1
                                    print("{0}/{1} Models evaluated".format(models_evaluated, grid_size))

