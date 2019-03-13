import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def separate_train():
    train_data_file = "data/train.csv"    
    if not os.path.isfile(train_data_file):
        print("train.csv file not found in data folder")
        print("this file is not commited in the repo and needs to be added manually")
        return (None, None, None, None)

    train_dest_file = "data/new_only_train.csv"
    val_dest_file = "data/new_only_val.csv"

    chunks_processed = 0
    data = None
    chunk_size = 150000
    train_sample_n = 3334   # We have about 629 million rows in total, so dedicating 3334 * 150 000  = 500 100 000 for training
    train_row_n = chunk_size * train_sample_n
    val_row_n = 129000000

    first_row = "\n"
    with open(train_data_file) as input_f:
        with open(train_dest_file, 'w') as dest:
            for i in range(train_row_n + 1):
                continue
                # if i == 0:
                #     first_row = input_f.readline()
                #     dest.write(first_row)
                #     continue

                # dest.write(input_f.readline())
                # if i % int(1e6) == 0:
                #     print("{0}/{1} train Lines written".format(i, train_row_n + 1)) 

        with open(val_dest_file, 'w') as dest:
            dest.write(first_row)
            for i in range(val_row_n):
                dest.write(input_f.readline())
                if i % int(1e6) == 0:
                    print("{0}/{1} val Lines written".format(i, val_row_n + 1)) 

def separate_train():
    train_data_file = "data/train.csv"    
    if not os.path.isfile(train_data_file):
        print("train.csv file not found in data folder")
        print("this file is not commited in the repo and needs to be added manually")
        return (None, None, None, None)

    train_dest_file = "data/new_only_train.csv"
    val_dest_file = "data/new_only_val.csv"

    chunks_processed = 0
    data = None
    chunk_size = 150000
    train_sample_n = 3334   # We have about 629 million rows in total, so dedicating 3334 * 150 000  = 500 100 000 for training
    train_row_n = chunk_size * train_sample_n
    val_row_n = 129000000

    first_row = "\n"
    with open(train_data_file) as input_f:
        with open(train_dest_file, 'w') as dest:
            for i in range(train_row_n + 1):
                continue
                # if i == 0:
                #     first_row = input_f.readline()
                #     dest.write(first_row)
                #     continue

                # dest.write(input_f.readline())
                # if i % int(1e6) == 0:
                #     print("{0}/{1} train Lines written".format(i, train_row_n + 1)) 

        with open(val_dest_file, 'w') as dest:
            dest.write(first_row)
            for i in range(val_row_n):
                dest.write(input_f.readline())
                if i % int(1e6) == 0:
                    print("{0}/{1} val Lines written".format(i, val_row_n + 1)) 

def separate_processed():
    x_data_file = "data/processed_x_train.csv"
    y_data_file = "data/processed_y_train.csv"


    x = np.loadtxt(x_data_file, delimiter=",")
    y = np.loadtxt(y_data_file, delimiter=",")

    print(x.shape)
    print(y.shape)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
    
    np.savetxt("data/new_x_train.csv", x_train, delimiter=",")
    np.savetxt("data/new_y_train.csv", y_train, delimiter=",")
    np.savetxt("data/new_x_val.csv", x_val, delimiter=",")
    np.savetxt("data/new_y_val.csv", y_val, delimiter=",")

separate_processed()
    

    
