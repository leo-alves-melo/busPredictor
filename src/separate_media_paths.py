# Separate the media paths to create a csv to train in a ML algorithm
# For each path, it will create a vector of x_length and the number of coordinate n path lenght 
# The training vector X is formed by x_length tuples of each coordinate, witch has the composition [(time, latitude, longitude)]
# For training porpouse, the vector will be something like [time_1, latitude_1, longitude_1, ... , time_x_length, latitude_x_length, longitude_x_length]

import pandas as pd
from lib.classes import *
from lib.data_filter import *
from lib.data_processor import *

x_length = 60

def calculate_total_distance(path_df):
	total_distance = 0
	last_row = path_df.iloc[0]
	for index, row in path_df.tail(-1).iterrows():
		total_distance += distance_between(last_row, row)
		last_row = row
	return total_distance

# Create the training path vector using the total distance as a divisor
def create_training_path(path_df):
	empty_coordinate = [0.0, 0.0, 0.0]
	total_distance = calculate_total_distance(path_df)
	path_size = len(path_df.index)
	cluster_distance = total_distance/path_size
	new_path = []
	path_index = 0

	for index in range(0, x_length):

		if path_index >= path_size:
			new_path += empty_coordinate
			continue

		current_distance = 0
		cluster = pd.DataFrame()
		last_row = path_df.iloc[path_index]
		cluster = cluster.append(last_row)
		path_index += 1

		if path_index >= path_size:
			new_path += [cluster.iloc[0].date, cluster.iloc[0].latitude, cluster.iloc[0].longitude]
			continue

		while current_distance < cluster_distance:
			current_row = path_df.iloc[path_index]
			current_distance += distance_between(last_row, current_row)
			cluster = cluster.append(current_row)
			path_index += 1
			last_row = current_row

			if path_index >= path_size:
				break

		mean_cluster = cluster.mean()
		cluster_coordinate = [mean_cluster.date, mean_cluster.latitude, mean_cluster.longitude]

		new_path += cluster_coordinate

	return new_path


paths_df = pd.read_csv('../data/media_paths_date.csv')

total = paths_df.index_path.max() + 1
count_int = 0

train_df = pd.DataFrame()
minimum_path = int(paths_df.index_path.min())
maximum_path = int(paths_df.index_path.max()) + 1

for path_id in range(minimum_path, maximum_path, 4):

	current_path_df = paths_df[paths_df.index_path == path_id]

	print 'path id: ' + str(path_id)

	line = current_path_df.iloc[0].id_line
	
	for lenght in range(1, len(current_path_df.index) + 1, 8):

		current_train_df = [create_training_path(current_path_df.head(lenght)) + [line]]

		train_df = train_df.append(current_train_df)

	if 100 * path_id / total > count_int:
		count_int += 1
		print count_int
		train_df.to_csv('../data/train_df.csv', index=False, header=False)

train_df.to_csv('../data/train_df.csv', index=False, header=False)




