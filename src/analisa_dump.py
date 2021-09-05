import pandas as pd
from math import sin, cos, sqrt, atan2, radians
import csv
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

class DumpAlgorithm():
	def fit(self, X, y):
		pass
	
	def predict(self, path):
		musium_hot_point = Coordinate(-22.814127, -47.058833)
		circular1_hot_point = Coordinate(-22.824493, -47.059594)
		fec_hot_point = Coordinate(-22.817975, -47.059667)

		if has_distance_from_coordinate(path, musium_hot_point, 0.3):
			return 4
		if has_distance_from_coordinate(path, circular1_hot_point, 0.25):
			return 1
		if has_distance_from_coordinate(path, fec_hot_point, 0.25):
			return 2

		return 3

class Coordinate():
	def __init__(self, latitude, longitude, timestamp = None, line_id = None, guid = None):
		self.latitude = latitude
		self.longitude = longitude
		self.timestamp = timestamp
		self.line_id = line_id
		self.device_guid = guid
	
	def from_coordinate(coordinate):
		return Coordinate(coordinate.latitude, coordinate.longitude, coordinate.timestamp, coordinate.line_id, coordinate.device_guid)

def has_distance_from_coordinate(path_df, coordinate, distance = 1):
	return (path_df.apply(lambda row: distance_between(Coordinate(row['latitude'], row['longitude']), coordinate) < distance, axis=1) == True).any()

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

def plot_path(path_df):

	latitudes = path_df.latitude
	longitudes = path_df.longitude
	
	numberOfDots = len(path_df.index)
	colors = cm.rainbow(np.linspace(0, 1, numberOfDots))
	
	plt.margins(0) # Fit the map in the image
	
	plt.scatter(longitudes, latitudes, color=colors)
	
	#plt.scatter(bus_stop_longitude, bus_stop_latitude, marker='*')
		
	plt.show()

paths_df = pd.read_csv('../data/media_paths_date.csv')

algorithm = DumpAlgorithm()

total = paths_df.index_path.max() + 1
count_int = 0

names = {1: 'Circular 1', 2: 'Circular 2 Fec', 3: 'Circular Noturno', 4: 'Circular 2 Museu'}

incorrect_index = []

with open('../data/incorrects_dump_humanos_vistos.csv', 'w') as incorrects_file:
	writer = csv.writer(incorrects_file)
	writer.writerow(['path id', 'predito', 'configurado', 'visto', 'tipo de erro'])
	humanos = [  729,   736,   925,   967,   981,   995,  1135,  1198,  1268,
        1324,  1450,  1590,  1758,  2731,  7015,  7141,  7148,  7155,
        7162,  7491,  7498,  7505, 12223, 13035, 13777, 14561, 14848,
       14911, 14967, 15044, 15072, 15149, 15247, 15275, 15401, 15436,
       15597, 15639, 15730, 15905, 16052, 16129, 16206, 16479, 16493,
       16500, 17116, 17137, 17228, 17312, 17319, 17326, 17333]
	for path_id in humanos:
		
		if 100 * path_id / total > count_int:
			count_int += 1
			print(count_int)
		current_path_df = paths_df[paths_df.index_path == path_id]
		predicted = algorithm.predict(current_path_df)
		if predicted != current_path_df.iloc[0].id_line:
			print("incorreto path:", path_id, "- com previsao:", names[predicted], "- e certo:", names[int(current_path_df.iloc[0].id_line)], '- hora:', current_path_df.iloc[0].date/60/60)
			
			plot_path(current_path_df)
			line = [path_id, names[predicted], names[int(current_path_df.iloc[0].id_line)], names[int(input())], 'humano']

			writer.writerow(line)



