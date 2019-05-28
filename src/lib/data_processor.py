# Data Processing Functions

import pandas as pd
from data_filter import *

def average_window(window):
	return window.mean()
	
def apply_average_window_on_path(path, window_size = 10, window_shift = 2):
	index = 0
	new_path = []
	while index < len(path):
		window = path[index:index + window_size]
		new_coordinate = average_window(window)
		new_path.append(new_coordinate)
		index += window_shift
		
	return new_path

def apply_average_time_window(path_df, window_time = 10, window_time_shift = 2):
	index = 0
	new_path_df = pd.DataFrame()
	initial_time = path_df.date.min()
	
	while index < len(path_df.index):
		
		window = pd.DataFrame()
		while index < len(path_df.index) and path_df.date.iloc[index] < initial_time + window_time:
			window = window.append(path_df.iloc[index])
			index += 1
			
		initial_time += window_time
		
		if len(window.index) > 0:
			average = average_window(window)
			
			new_path_df = new_path_df.append(average, ignore_index=True)
		
	return new_path_df

def separate_X_coordinate(coordinate, first_time):
	current_time = collect_time(coordinate.timestamp)
	difference_time = (current_time - first_time).total_seconds() 
	return [difference_time, coordinate.latitude, coordinate.longitude]

def separate_Y_id_line(path):
	return path[0].id_line

def separate_X_paths(paths):
	X = []
	for path in paths:
		first_time = collect_time(path[0].timestamp)
		X.append(list(map(lambda coordinate: separate_X_coordinate(coordinate, first_time), path)))
		
	return X

def separate_Y_paths(paths):
	Y = []
	for path in paths:
		Y.append(separate_Y_id_line(path))
		
	return Y     

def separate_train_test(paths):
	all_X = separate_X_paths(paths) 
	all_y = separate_Y_paths(paths)
	
	X_train, X_test, y_train, y_test = train_test_split(all_X, all_y, random_state=1, test_size=0.2, stratify=all_y)
	
	return X_train, X_test, y_train, y_test