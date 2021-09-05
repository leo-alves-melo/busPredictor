import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from math import sin, cos, sqrt, atan2, radians
from joblib import dump, load
import numpy as np
import sys
import math

def distance_between(position1, position2):
	# approximate radius of earth in km
	R = 6373.0

	lat1 = radians(position1.latitude)
	lon1 = radians(position1.longitude)
	lat2 = radians(position2.latitude)
	lon2 = radians(position2.longitude)

	dlon = lon2 - lon1
	dlat = lat2 - lat1
	
	a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
	c = 2 * atan2(sqrt(a), sqrt(1 - a))
	
	distance = R * c
	
	return distance

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

media_path_location = '../data/media_paths_date.csv'

train_location = '../data/'

paths_df = pd.read_csv(media_path_location)

lenght_list = [90]

for x_length in lenght_list:

    print('lenght:', x_length)
    total = paths_df.index_path.max() + 1
    count_int = 0

    train_df = pd.DataFrame()
    minimum_path = int(paths_df.index_path.min())
    maximum_path = int(paths_df.index_path.max()) + 1

    for path_id in range(minimum_path, maximum_path, 4):

        current_path_df = paths_df[paths_df.index_path == path_id]

        line = current_path_df.iloc[0].id_line
        
        for lenght in range(1, len(current_path_df.index) + 1, 8):

            current_train_df = [create_training_path(current_path_df.head(lenght)) + [line]]

            train_df = train_df.append(current_train_df)

        if 100 * path_id / total > count_int:
            count_int += 1
            print(count_int)
            train_df.to_csv(train_location + 'train_df_' + str(x_length) + '.csv', index=False, header=False)

    train_df.to_csv(train_location + 'train_df_' + str(x_length) + '.csv', index=False, header=False)