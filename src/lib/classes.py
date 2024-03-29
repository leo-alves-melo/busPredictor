# Classes definitions

from lib.constants import *

# Coordinate of a 
class Coordinate():
	def __init__(self, latitude, longitude, timestamp = None, line_id = None, guid = None):
		self.latitude = latitude
		self.longitude = longitude
		self.timestamp = timestamp
		self.line_id = line_id
		self.device_guid = guid
	
	def from_coordinate(coordinate):
		return Coordinate(coordinate.latitude, coordinate.longitude, coordinate.timestamp, coordinate.line_id, coordinate.device_guid)

home = Coordinate(latitude_home, longitude_home)

class Path():
	def __init__(self, coordinates, line_id):
		self.coordinates = coordinates
		self.line_id = line_id
		
class Predictor():
	def __init__(self, algorithm):
		self.algorithm = algorithm
		self.next_path()
		
	def train(self, X, y):
		self.algorithm.fit(np.array(X), np.array(y))
	
	def next_path(self):
		self.reset_path()
		
	def reset_path(self):
		self.path = []
	
	def iterate(self, coordinate):
		self.path.append(coordinate)
		return self.algorithm.predict(np.array(self.path))
	
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

class SimilarityAlgorithm():

	def __init__(self, window_size, paths_models, similarity_threshold):
		self.window_size = window_size
		self.paths_models = paths_models
		self.similarity_threshold = similarity_threshold**2

	def fit(self, X, y):
		pass

	def similarity(self, path, model):
		neighboors = 0
		for index in range(len(path.index) - self.window_size):
			current_similarity = 0
			for window_index in range(self.window_size):
				current_similarity += (path.iloc[index + window_index].latiude - model.iloc[window_index].latitude)**2 + (path.iloc[index + window_index].longitude - model.iloc[window_index].longitude)**2
			if current_similarity < self.similarity_threshold:
				neighboors += 1
		
		return neighboors/(len(path.index) - self.window_size)
	
	def predict(self, path):
		best_model = None
		best_similarity = 0
		for model in self.paths_models:
			current_similarity = self.similarity(path, self.paths_models[model])
			if current_similarity > best_similarity:
				best_similarity = current_similarity
				best_model = model

		return best_model

# Bayes algorithm fits the model for 350 positions, 
# which means almost 1h oh bus travel
# Max size of all columns
max_column = 350
class BayesAlgorithm():
	def fit(self, X, y):

		train = []
		response = []
		
		for index_line in range(len(X)):
			for index_column in range(max_column):
				
				train_sample = X[index_line][0:index_column+1]
				train_sample = np.array(reduce(list.__add__, train_sample))
				appended = np.zeros(3*max_column - len(train_sample))
				
				train_sample = np.append(train_sample, appended)
				
				train.append(np.array(train_sample))
				response.append(y[index_line])
		
		train = np.array(train)
		response = np.array(response)
		
		self.model = GaussianNB().fit(pd.DataFrame(train), pd.DataFrame(response))
	
	def predict(self, path):
		
		train_sample = path.flatten()
		appended = np.zeros(3*max_column - len(train_sample))

		sample = np.append(np.array(path), appended)
		return self.model.predict(np.array(sample).reshape(1, -1))[0]
		