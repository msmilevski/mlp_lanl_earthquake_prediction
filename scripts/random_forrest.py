import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import data_provider

x_train, y_train, x_val, y_val = data_provider.get_data()

rnd_seed = 123131

# # don't know how to use criterion as in the paper
# As in the paper
# regressor = RandomForestRegressor(n_estimators=1000, random_state=rnd_seed,
#     min_samples_split=30, min_samples_leaf=30, max_features=40)

regressor = RandomForestRegressor(n_estimators=1000, random_state=rnd_seed,
    min_samples_split=30, min_samples_leaf=30, max_features=28)

regressor.fit(x_train, y_train)

predictions = regressor.predict(x_val)
error = mean_absolute_error(y_val, predictions)

print(error)