import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import data_provider
import sys

run_nr = int(np.random.rand() * 10**6)
if len(sys.argv) > 1:
	run_nr = sys.argv[1]

base_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_path = os.path.join(base_dir_path, "results", "random_forrest", "random_forrest_new_{0}.csv".format(run_nr))

def model_name(n_est, min_samples_split, min_samples_leaf, max_features):
	return "n_estimators: {0}, min_samples_split: {1}, min_samples_leaf: {2}, max_features:{3}".format(
		n_est, min_samples_split, min_samples_leaf, max_features)

x_train, y_train, x_val, y_val = data_provider.get_data()

def write_to_results(text, mode="a"):
	with open(results_path, mode) as f:
		f.write(text + "\n")

def save_result(error, n_est, min_split, min_leaf, max_feat):
	write_to_results("{0},{1},{2},{3},{4}".format(n_est, min_split, min_leaf, max_feat, error))

def save_result_with_error_bar(mean, error_bar, n_est, min_split, min_leaf, max_feat):
	write_to_results("{0},{1},{2},{3},{4},{5}".format(n_est, min_split, min_leaf, max_feat, mean, error_bar))	

write_to_results("n_est,min_samples_split,min_samples_leaf,max_features", mode="w")

def evaluate_model(n_estimators, min_samples_split, min_samples_leaf, max_features, rnd_seed=123131):
	current_model_name = model_name(n_estimators, min_samples_split, min_samples_leaf, max_features)
	print("Training model:")
	print(current_model_name)

	regressor = RandomForestRegressor(n_estimators=n_estimators, random_state=rnd_seed,
	    min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, 
	    max_features=max_features)
	
	regressor.fit(x_train, y_train)

	train_predictions = regressor.predict(x_train)
	train_error = mean_absolute_error(y_train, train_predictions)	
	print("train error: {0}".format(train_error))

	val_predictions = regressor.predict(x_val)
	val_error = mean_absolute_error(y_val, val_predictions)
	print("val error: {0}".format(val_error))

	return val_error

def grid_search():
	n_estimators_grid = [750, 1250]
	min_samples_split_grid = [2]
	min_samples_leaf_grid = [75, 100, 125, 150, 175]
	max_features_grid = [2, 3, 4, 15, 30, 45]
	grid_size = len(n_estimators_grid) * len(min_samples_split_grid) \
		* len(min_samples_leaf_grid) * len(max_features_grid)

	models_evaluated = 0
	for n_estimators in n_estimators_grid:
		for min_samples_split in min_samples_split_grid:
			for min_samples_leaf in min_samples_leaf_grid:
				for max_features in max_features_grid:
					error = evaluate_model(n_estimators, min_samples_split, min_samples_leaf, max_features)
					save_result(error, n_estimators, min_samples_split, min_samples_leaf, max_features)	

					models_evaluated += 1
					print("{0}/{1} Models evaluated".format(models_evaluated, grid_size))

def get_error_bar(errors):
	errors = np.array(errors)
	mean = np.mean(errors)
	std = np.std(errors)
	n = len(errors)

	# print("-----TESTTTT-----")
	# print(errors)
	# print(mean)
	# print(n)

	return mean, std / np.sqrt(n)


def best_model_evaluation():
	best_models = [
		(500, 2, 200, 2),
		(400, 2, 175, 2),
		(600, 2, 175, 2),
		# (1250, 2, 600, 4),
		(500, 2, 200, 2),
		# (1250, 2, 125, 2),
		# (1000, 2, 150, 2),
		# (750, 2, 150, 2),
		# (750, 2, 125, 2)
	]

	rnd_seeds = [123131]#, 856856, 293420, 775241, 5562960]

	for n_estimators, min_samples_split, min_samples_leaf, max_features in best_models:		
		errors = []
		for rnd_seed in rnd_seeds:
			error = evaluate_model(n_estimators, min_samples_split, min_samples_leaf, 
				max_features, rnd_seed=rnd_seed)
			errors += [error]

		mean, error_bar = get_error_bar(errors)
		save_result_with_error_bar(mean, error_bar, n_estimators, min_samples_split,
			min_samples_leaf, max_features)

def evaluate_test_set():
	submission = pd.read_csv('data/sample_submission.csv')

	x_test = data_provider.get_test_x() 
	regressor = RandomForestRegressor(n_estimators=1000, random_state=123131,
	    min_samples_split=2, min_samples_leaf=100, 
	    max_features=2)
	
	regressor.fit(x_train, y_train)

	train_predictions = regressor.predict(x_train)
	train_error = mean_absolute_error(y_train, train_predictions)	
	print("train error: {0}".format(train_error))

	test_predictions = regressor.predict(x_test)

	print(test_predictions.shape)
	print(submission.time_to_failure.shape)

	submission['time_to_failure'] = test_predictions

	submission.to_csv("submission.csv", index=False)

best_model_evaluation()










