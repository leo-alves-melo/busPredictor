import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from joblib import dump, load
import numpy as np
import sys
from statsmodels.tsa.vector_ar.var_model import VAR
import math
from math import sin, cos, sqrt, atan2, radians

x_length = 60

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


# Calculates the distance in km
def distance_between_literal(lat1, lon1, lat2, lon2):
	# approximate radius of earth in km
	R = 6373.0

	lat1 = radians(lat1)
	lon1 = radians(lon1)
	lat2 = radians(lat2)
	lon2 = radians(lon2)

	dlon = lon2 - lon1
	dlat = lat2 - lat1
	
	a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
	c = 2 * atan2(sqrt(a), sqrt(1 - a))
	
	distance = R * c
	
	return distance

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

train_location = '../data/'


train_df = pd.read_csv(train_location + 'train_df_60.csv', header=None)
paths_df = pd.read_csv(train_location + 'media_paths_date.csv')

paths_train_df = paths_df[paths_df.index_path % 4 == 0]
paths_test_df = paths_df[paths_df.index_path % 4 == 1]

path_1_df = paths_train_df[paths_train_df.id_line == 1]
path_1_df = path_1_df.drop(['id_line', 'index_path'], axis=1)
model_1 = VAR(path_1_df).fit()

path_2_df = paths_train_df[paths_train_df.id_line == 2]
path_2_df = path_2_df.drop(['id_line', 'index_path'], axis=1)
model_2 = VAR(path_2_df).fit()

path_3_df = paths_train_df[paths_train_df.id_line == 3]
path_3_df = path_3_df.drop(['id_line', 'index_path'], axis=1)
model_3 = VAR(path_3_df).fit()

path_4_df = paths_train_df[paths_train_df.id_line == 4]
path_4_df = path_4_df.drop(['id_line', 'index_path'], axis=1)
model_4 = VAR(path_4_df).fit()

index_paths = paths_test_df.index_path.unique()
maximum_path = int(paths_test_df.index_path.max()) + 1
count_int = 0
new_df = pd.DataFrame()
for index_path in index_paths:

	lenght = 0
	single_test_df = paths_test_df[paths_test_df.index_path == index_path]
	id_line = single_test_df.id_line.iloc[0]
	processed_test_df = single_test_df.drop(['id_line', 'index_path'], axis=1)

	print('path_id:', index_path, end=" - ")

	total_dist_1 = 0
	total_dist_2 = 0
	total_dist_3 = 0
	total_dist_4 = 0
	for index in range(1, len(processed_test_df.index) - 1, 8):
		route = processed_test_df.head(index)
		next_line = np.array(processed_test_df.iloc[index + 1])

		array_route = np.array(route)
		prediction_1 = model_1.forecast(array_route, steps=1)[0]
		prediction_2 = model_2.forecast(array_route, steps=1)[0]
		prediction_3 = model_3.forecast(array_route, steps=1)[0]
		prediction_4 = model_4.forecast(array_route, steps=1)[0]

		# Define closest prediction
		dist_1 = distance_between_literal(prediction_1[1], prediction_1[2], next_line[1], next_line[2])
		dist_2 = distance_between_literal(prediction_2[1], prediction_2[2], next_line[1], next_line[2])
		dist_3 = distance_between_literal(prediction_3[1], prediction_3[2], next_line[1], next_line[2])
		dist_4 = distance_between_literal(prediction_4[1], prediction_4[2], next_line[1], next_line[2])

		# Use time as distance too, centering using the value of the day
		next_line_time = next_line[0]/(24*60*60)**2
		dist_1 = math.sqrt((prediction_1[0]/(24*60*60) - next_line_time)**2 + dist_1**2)
		dist_2 = math.sqrt((prediction_2[0]/(24*60*60) - next_line_time)**2 + dist_2**2)
		dist_3 = math.sqrt((prediction_3[0]/(24*60*60) - next_line_time)**2 + dist_3**2)
		dist_4 = math.sqrt((prediction_4[0]/(24*60*60) - next_line_time)**2 + dist_4**2)

		total_dist_1 += dist_1
		total_dist_2 += dist_2
		total_dist_3 += dist_3
		total_dist_4 += dist_4

		entry = np.concatenate((prediction_1, prediction_2, prediction_3, prediction_4, np.array([total_dist_1, total_dist_2, total_dist_3, total_dist_4]), create_training_path(route), np.array([id_line])))
		entry_df = pd.DataFrame(np.array([entry]))

		new_df = new_df.append(entry_df, ignore_index=True)

		#predicted_timeSeries = values.index(min(values)) + 1

		# Parse to percent 
		#lenght = int(100*index/len(processed_test_df.index) + 1)

		# Append the new number of total of this lenght and add if it was correct
		#correctness['total_' + str(int(id_line)) + '_' + str(lenght)] += 1

		#if predicted_timeSeries == id_line:
		#    correctness['correct_timeSeries_' + str(int(id_line)) + '_' + str(lenght)] += 1
	
	if 100 * index_path / maximum_path > count_int:
		count_int += 1
		print('\n' + str(count_int))
		new_df.to_csv(train_location + 'train_df_times_seriesFull.csv', index=False, header=False)

