import pandas as pd

def iterate_test_data():
	sample_submission = pd.read_csv("data/sample_submission.csv")
	for seg_id in sample_submission.seg_id.values:
		data = pd.read_csv("data/test/{0}.csv".format(seg_id))
		yield data


