from keras.models import Sequential
import keras
from keras.utils import to_categorical
from keras.layers import Dense, Conv1D, Conv2D, Flatten, Reshape, MaxPooling1D, MaxPooling2D, GlobalAveragePooling1D, Dropout, BatchNormalization
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump, load
import numpy as np

# Normalizing X considering the absolute values of the coordinates
highest_latitude = 22.836006
lowest_latitude = 22.801396
highest_longitude = 47.095658
lowest_longitude = 47.046078
hightest_time = 86400
lowest_time = 0

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

def adjustTimeColumn(time):
	if time == 0:
		return 0
	return (time - lowest_time)/(hightest_time - lowest_time)

def adjustLatitudeColumn(latitude):
	if latitude == 0:
		return 0
	return (latitude - lowest_latitude)/(highest_latitude - lowest_latitude)

def adjustLongitudeColumn(longitude):
	if longitude == 0:
		return 0
	return (longitude - lowest_longitude)/(highest_longitude - lowest_longitude)

def adjustTrainDf(df):

	for column in df:
		# For time columns
		if column % 3 == 0:
			df[column] = df[column].apply(adjustTimeColumn)
		elif column % 3 == 1:
			df[column] = df[column].apply(adjustLatitudeColumn)
		elif column % 3 == 2:
			df[column] = df[column].apply(adjustLongitudeColumn)

	return df

def convertTo3D(df):
	time_train_df = []
	lat_train_df = []
	lon_train_df = []
	for line in range(len(df.index)):

		time_train_df.append(df.iloc[line][df.iloc[line].index % 3 == 0].to_numpy())
		lat_train_df.append(df.iloc[line][df.iloc[line].index % 3 == 1].to_numpy())
		lon_train_df.append(df.iloc[line][df.iloc[line].index % 3 == 2].to_numpy())
	new_df = np.array([np.array(time_train_df), np.array(lat_train_df), np.array(lon_train_df)])
	return new_df.reshape((new_df.shape[1], 60, 3))

# Constants and global variables

best_nn_location = '../data/best_nn_cross_rio_bus_data.h5'

train_df = []

print("lendo dados...")

y_df = []
X_df = []

for index in range(2):
	print('index:', index)
	y_df.append(pd.read_csv('../data/3d_riobus_' + str(index) + '_y.csv', header=None))
	current_x = pd.read_csv('../data/3d_riobus_' + str(index) + '_x.csv', header=None)
	X_df.append(pd.DataFrame(current_x.to_numpy().reshape((current_x.shape[0], 60, 3))))
	
print("Criando df...")
print(pd)
X_df = pd.concat(X_df, axis=0, ignore_index=True)
y_df = pd.concat(y_df, axis=0, ignore_index=True)

print("dados lidos!")

X_train_all, X_test, y_train_all, y_test = train_test_split(X_df, y_df, random_state=1, test_size=0.2, stratify=y_df)

#X_train_all = adjustTrainDf(X_train_all.abs())

X_train, X_cross, y_train, y_cross = train_test_split(X_train_all, y_train_all, random_state=1, test_size=0.2, stratify=y_train_all)

#X_test = adjustTrainDf(X_test.abs())

y_train = to_categorical(y_train.apply(lambda x: x-1))
y_cross = to_categorical(y_cross.apply(lambda x: x-1))
y_test = to_categorical(y_test.apply(lambda x: x-1))

X_train = convertTo3D(X_train)
X_cross = convertTo3D(X_cross)

print("treinando...")

model = Sequential()

model.add(Conv1D(64, 3, activation='relu', input_shape=(60,3)))

model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(128, 3, activation='relu'))

model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())

model.add(Dense(2000, activation='sigmoid'))
model.add(Dense(2000, activation='sigmoid'))
model.add(Dense(4, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, batch_size=500, epochs=50, validation_split=0.2, verbose=1)
acc = model.evaluate(X_cross, y_cross)[1]

print('acc: ' + str(acc))

model.save(best_nn_location)