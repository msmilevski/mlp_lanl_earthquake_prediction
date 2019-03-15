from overlapped_data_provider import OverlappedDataProvider
from overlapped_data_provider2 import OverlappedDataProvider2 as ODP
import numpy as np

out_path = "dp_test_out2.out"
def write_to_out(text, mode="a"):
	with open(out_path, mode) as f:
		f.write(text + "\n")

def test_1():
	np.set_printoptions(precision=10)
	dp = OverlappedDataProvider(data_filepath='data/provider_test.csv', batch_size=3, chunk_size=30, num_chunks=2)
	x1, y1, x2, y2 = [], [], [], []	
	for indx, (x, y) in enumerate(dp):
		x1.append(x)		
		y1.append(y)
		if indx < 3:
			print(x)
			print(y)

	print("\n\n")
	for indx, (x, y) in enumerate(dp):
		x2.append(x)
		y2.append(y)
		if indx < 3:
			print(x)
			print(y)

	print(np.allclose(np.array(x1), np.array(x2)))
	print(np.allclose(np.array(y1), np.array(y2)))


def test_2():
	dp = OverlappedDataProvider(data_filepath='data/only_train.csv', batch_size=1)
	for indx, (x, y) in enumerate(dp):
		print("um what?")
		write_to_out("x {0}:".format(indx))

		for i in range(x.shape[1]):
			write_to_out(str(x[0, i]))

		write_to_out("\ny{0}:")
		write_to_out(str(y[0]) + "\n")

		if indx > 5:
			break

def test_3():
	dp = ODP(batch_size=3, segment_size=5, overlap_fraction=0.4, file_path='data/provider_test_2.csv', data_splits=2)
	np.set_printoptions(precision=10)
	for indx, (x, y) in enumerate(dp):
		print(x)
		print(y)

test_3()