import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import data_provider

base_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_path = os.path.join(base_dir_path, "results", "random_forrest.csv")

def model_name(n_est, min_samples_split, min_samples_leaf, max_features):
	return "n_estimators: {0}, min_samples_split: {1}, min_samples_leaf: {2}, max_features:{3}".format(
		n_est, min_samples_split, min_samples_leaf, max_features)

x_train, y_train, x_val, y_val = data_provider.get_data()

rnd_seed = 123131

n_estimators_grid = [100, 500, 1000]
min_samples_split_grid = [4, 10, 30]
min_samples_leaf_grid = [4, 15, 30]
max_features_grid = [10]   # [10, 30, 50]
grid_size = len(n_estimators_grid) * len(min_samples_split_grid) \
	* len(min_samples_leaf_grid) * len(max_features_grid)

# # don't know how to use criterion as in the paper
# As in the paper
# regressor = RandomForestRegressor(n_estimators=1000, random_state=rnd_seed,
#     min_samples_split=30, min_samples_leaf=30, max_features=40)

# regressor = RandomForestRegressor(n_estimators=1000, random_state=rnd_seed,
#     min_samples_split=30, min_samples_leaf=30, max_features=28)

def write_to_results(text, mode="a"):
	with open(results_path, mode) as f:
		f.write(text + "\n")

def save_result(error, n_est, min_split, min_leaf, max_feat):
	write_to_results("{0},{1},{2},{3},{4}".format(n_est, min_split, min_leaf, max_feat, error))

write_to_results("n_est,min_samples_split,min_samples_leaf,max_features", mode="w")


models_evaluated = 0
for n_estimators in n_estimators_grid:
	for min_samples_split in min_samples_split_grid:
		for min_samples_leaf in min_samples_leaf_grid:
			for max_features in max_features_grid:
				current_model_name = model_name(n_estimators, min_samples_split, min_samples_leaf, max_features)
				print("Training model:")
				print(current_model_name)

				regressor = RandomForestRegressor(n_estimators=n_estimators, random_state=rnd_seed,
				    min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, 
				    max_features=max_features)

				regressor.fit(x_train, y_train)

				predictions = regressor.predict(x_val)
				error = mean_absolute_error(y_val, predictions)

				print(error)

				save_result(error, n_estimators, min_samples_split, min_samples_leaf, max_features)

				models_evaluated += 1
				print("{0}/{1} Models evaluated".format(models_evaluated, grid_size))

