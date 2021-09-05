import pandas as pd
#from classes import *
#from lib.data_filter import *

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
				current_similarity += (path.iloc[index + window_index].latitude - model.iloc[window_index].latitude)**2 + (path.iloc[index + window_index].longitude - model.iloc[window_index].longitude)**2
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

paths_df = pd.read_csv('../data/media_paths_date.csv')

model = SimilarityAlgorithm(window_size=10, paths_models={1: paths_df[paths_df.index_path == 1127.0], 2: paths_df[paths_df.index_path == 1], 3: paths_df[paths_df.index_path == 724], 4: paths_df[paths_df.index_path == 2766]}, similarity_threshold=0.001)

print(model.predict(paths_df[paths_df.index_path == 1]))

