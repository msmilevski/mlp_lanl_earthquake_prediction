import pandas as pd

raw_data_file = "/Users/dziugas/Documents/code/mlp/lanl_earthquake/data/train.csv"
train_output_file = "/Users/dziugas/Documents/code/mlp/lanl_earthquake/data/downsampled4_train.csv"
val_output_file = "/Users/dziugas/Documents/code/mlp/lanl_earthquake/data/downsampled4_val.csv"

with open(train_output_file, 'w') as f:
	f.write("acoustic_data,time_to_failure\n")

with open(val_output_file, 'w') as f:
	f.write("acoustic_data,time_to_failure\n")


train_sample_n = 3334
sample_size = 150000

chunks_processed = 0
for chunk in pd.read_csv(raw_data_file, chunksize=sample_size):
    if chunk.shape[0] < sample_size:        # throwing out last remaining rows
        break

    values = chunk.values
    downsample_rate = 4

    downsampled = values.reshape(sample_size // downsample_rate, downsample_rate, 2).mean(axis=1)

    output_file = train_output_file if chunks_processed < train_sample_n else val_output_file 

    with open(output_file, 'a') as f:
        for row in downsampled:
            f.write("{0},{1}\n".format(row[0], row[1]))

    chunks_processed += 1
    print("{0}/{1} chunks processed".format(chunks_processed, 4193))
