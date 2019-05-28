import pandas as pd
from lib.classes import *
from lib.data_filter import *
from lib.classes import *

paths_df = pd.read_csv('../data/paths.csv')

algorithm = DumpAlgorithm()

total = paths_df.index_path.max() + 1
count_int = 0

predicted = []
correct = []

for path_id in range(paths_df.index_path.min(), paths_df.index_path.max() + 1):
	
	if 100 * path_id / total > count_int:
		count_int += 1
		print count_int
	current_path_df = paths_df[paths_df.index_path == path_id]
	
	predicted.append(algorithm.predict(current_path_df))
	correct.append(int(current_path_df.id_line.iloc[0]))

predict_df = pd.DataFrame(list(zip(predicted, correct)))
predict_df.to_csv('../data/predicted_dump.csv', index=False)
