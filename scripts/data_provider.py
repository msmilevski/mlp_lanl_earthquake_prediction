import pandas as pd

def get_data():
	x_train = pd.read_csv("data/x_train.csv")
	y_train = pd.read_csv("data/y_train.csv")
	x_val = pd.read_csv("data/x_val.csv")
	y_val = pd.read_csv("data/y_val.csv")

	return (x_train, y_train, x_val, y_val)