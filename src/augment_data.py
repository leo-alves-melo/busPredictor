import pandas as pd
import numpy as np

media_path_location = '../data/train_df.csv'

train_location = '../data/'

new_train_df = pd.read_csv(media_path_location)
model_train_df = new_train_df.copy()

model_train_df.to_csv('../data/augmented_train_0.csv', index=False)

multiply = 50

current_index_path = paths_df.index_path.max() + 1

for mult in range(1, multiply + 1):

	current_df = model_train_df.copy()
	size = len(current_df.index)

	for column in range(180):
		if column % 3 == 0:
			current_df[column] = current_df[column] + ((current_df[column] != 0)*np.random.normal(0, 60, size))
		if column % 3 == 1:
			current_df[column] = current_df[column] + ((current_df[column] != 0)*np.random.normal(0, 0.00005, size))
		if column % 3 == 2:
			current_df[column] = current_df[column] + ((current_df[column] != 0)*np.random.normal(0, 0.0001, size))

	current_df.to_csv('../data/augmented_train_' + str(mult) + '.csv', index=False, header=False)