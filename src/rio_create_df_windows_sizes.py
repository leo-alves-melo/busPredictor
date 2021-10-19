import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from math import sin, cos, sqrt, atan2, radians
from joblib import dump, load
import numpy as np
import sys
import math
from lib.rio_data_filter import *
from lib.training_path import *

small_riobus = '../data/rio_paths_10.csv'
train_location = '../data/'

paths_df = pd.read_csv(small_riobus, parse_dates=['datetime'], dtype={'line': object})

paths_df.datetime = ((paths_df.datetime - paths_df.datetime.dt.normalize()) / pd.Timedelta('1 second')).astype(int)
paths_df = paths_df.drop(columns=['order'])

lenght_list = [60]

for x_length in lenght_list:

	print('lenght:', x_length)

	lines = paths_df.line.unique()
	jump = 64

	train_df = pd.DataFrame()

	for line in lines:

		print('line:', line)
		current_df = paths_df[paths_df.line == line]

		total = current_df.index_path.max() + 1
		count_int = 0

		possible_index_paths = current_df.index_path.unique()
		total = possible_index_paths[-1] - possible_index_paths[0]

		for index_path in range(possible_index_paths[0], possible_index_paths[-1], jump):

			current_path_df = current_df[current_df.index_path == index_path]

			line = current_path_df.iloc[0].line
			
			for lenght in range(1, len(current_path_df.index) + 1, 4):

				current_train_df = [create_training_path(current_path_df.head(lenght)) + [line]]

				train_df = train_df.append(current_train_df)

			if 100 * index_path / total > count_int:
				count_int += 1
				print(count_int)
				train_df.to_csv(train_location + 'rio_train_df_' + str(x_length) + '.csv', index=False, header=False)

		train_df.to_csv(train_location + 'rio_train_df_' + str(x_length) + '.csv', index=False, header=False)