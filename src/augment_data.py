import pandas as pd
import numpy as np

media_path_location = '../data/media_paths_date.csv'

train_location = '../data/'

paths_df = pd.read_csv(media_path_location)
new_train_df = paths_df[paths_df.index_path % 4 == 0].copy()
model_train_df = new_train_df.copy()

model_train_df.to_parquet('../data/augmented_media_paths_date_0.parquet.gzip', compression='gzip')

multiply = 2000

current_index_path = paths_df.index_path.max() + 1

for mult in range(1, multiply + 1):

	if mult % 100 == 0:
		print(mult)

	current_df = model_train_df.copy()
	size = len(current_df.index)

	current_df.date += np.random.normal(0, 60, size)
	current_df.latitude += np.random.normal(0, 0.00005, size)
	current_df.longitude += np.random.normal(0, 0.0001, size)

	current_df.index_path += (mult*current_df.index_path.max())

	current_df.to_parquet('../data/augmented_media_paths_date_' + str(mult) + '.parquet.gzip', compression='gzip')