import pandas as pd

def get_data():
	x_train = pd.read_csv("data/new_x_train.csv")
	y_train = pd.read_csv("data/new_y_train.csv", header=None)
	x_val = pd.read_csv("data/new_x_val.csv")
	y_val = pd.read_csv("data/new_y_val.csv", header=None)

	return (x_train, y_train, x_val, y_val)

def get_test_x():
	return pd.read_csv("data/new_x_test.csv", header=None)