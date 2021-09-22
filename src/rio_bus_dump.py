# encoding: utf-8

from matplotlib.pyplot import axis
from pandas.core.frame import DataFrame
from lib.classes import *
from lib.data_visualization import *
from lib.rio_data_filter import *
import pandas as pd
import matplotlib.pyplot as plt

def test_dump(df, model):

	# test will be indexed in 3, 7, 11...

	lines = df.line.unique()
	jump = 64

	correct = {}
	total_try = {}

	for line in lines:
		print('line:', line)
		current_df = df[df.line == line]

		possible_index_paths = current_df.index_path.unique()
		total = possible_index_paths[-1] - possible_index_paths[0]
		count = 0
		for index_path in range(possible_index_paths[3], possible_index_paths[-1], jump):
			
			count += (1*jump)
			current_path = current_df[current_df.index_path == index_path]
			
			path_lenght = len(current_path.index)

			for lenght in range(0, path_lenght, 4):

				predicted = model.predict(current_path.head(lenght))

				if predicted == line:
					if correct.get(int(100*lenght/path_lenght)) == None:
						correct[int(100*lenght/path_lenght)] = 1
					else:
						correct[int(100*lenght/path_lenght)] += 1
				else:
					if correct.get(int(100*lenght/path_lenght)) == None:
						correct[int(100*lenght/path_lenght)] = 0	
				
				if total_try.get(int(100*lenght/path_lenght)) == None:
					total_try[int(100*lenght/path_lenght)] = 1
				else:
					total_try[int(100*lenght/path_lenght)] += 1
				

			print(count/total *100, '- predicted:', predicted, '- correct:', correct, '- tries:', total_try)
	
	return (correct, total_try)


def separate_go_back(df):
	file = open('../data/riobus_coordinates.json')
	data = json.load(file)

	final_stops = {}
	for key in data.keys():
		final_stops[key] = [Coordinate(latitude=data[key][0][0], longitude=data[key][0][1]), Coordinate(latitude=data[key][1][0], longitude=data[key][1][1])]
	df.line = df.apply(lambda x: x.line + '_ida' if distance_between(df[df.index_path == x.index_path].iloc[0], final_stops[x.line][0]) < 0.1 else x.line + '_volta', axis=1)
	return df

def create_paths_models(df):

	# Models will be indexed in 0, 64, 128...

	lines = df.line.unique()
	jump = 64

	new_df = DataFrame()

	for line in lines:
		print('line:', line)
		current_df = df[df.line == line]

		new_df = new_df.append(current_df.iloc[0])

		possible_index_paths = current_df.index_path.unique()
		total = possible_index_paths[-1] - possible_index_paths[0]
		count = 0
		for index_path in range(possible_index_paths[0], possible_index_paths[-1], jump):
			
			count += (1*jump)
			current_path = current_df[current_df.index_path == index_path]
			print(count/total *100)
			for row in current_path.iterrows():
				row = row[1]
				if not has_distance_from_coordinate(new_df[new_df.line == line], row, distance=minimum_distance):
					new_df = new_df.append(row)
					

	return new_df.drop(['index_path', 'order'], axis=1)
		

class DumpRioBusAlgorithm():
	def __init__(self, paths):
		self.paths = paths
		self.possible_lines = paths.line.unique()

	def fit(self, X, y):
		pass

	def predict(self, path):
		for row in path.iterrows():

			current_not_possible_lines = []

			row = row[1]

			for line in self.possible_lines:
				if not has_distance_from_coordinate(self.paths[self.paths.line == line], row, distance=0.1):
					current_not_possible_lines.append(line)

			if len(current_not_possible_lines) < len(self.possible_lines):
				self.possible_lines = list(set(self.possible_lines) - set(current_not_possible_lines))
			
			if len(self.possible_lines) == 1:
				break
		
		choosed_line = self.possible_lines[0]

		self.possible_lines = self.paths.line.unique()
		return choosed_line

def plot_dump_results(corrects, tries):
	result = [0]*101
	result[100] = 88.6 # This result was computed with completed paths
	for index in range(100):
		result[index] = 100*corrects[index]/tries[index]

	plt.plot(result, label=u'Georreferência Automático', marker='', markerfacecolor='blue')
	
	plt.xlabel('Porcentagem de Completude')
	plt.ylabel(u'Acurácia')
	plt.title(u'Porcentagem de Completude X Acurácia do Trajeto')
	plt.legend(loc='best')
	plt.grid(True)

	plt.show()


small_riobus = '../data/rio_paths_10.csv'
minimum_percent = 0.1
minimum_distance = 0.1

#models_df = pd.read_csv('../data/dump_models.csv', parse_dates=['datetime'], dtype={'line': object})
#df = pd.read_csv(small_riobus, parse_dates=['datetime'], dtype={'line': object})

#model = DumpRioBusAlgorithm(models_df)

corrects = {0: 452, 4: 90, 8: 139, 12: 176, 16: 193, 20: 247, 25: 276, 29: 199, 33: 233, 37: 207, 41: 263, 45: 230, 50: 371, 54: 248, 58: 250, 62: 255, 66: 367, 70: 299, 75: 342, 79: 77, 83: 267, 87: 228, 91: 225, 95: 229, 13: 182, 17: 191, 22: 195, 26: 209, 31: 163, 35: 241, 40: 319, 44: 238, 49: 112, 53: 252, 67: 185, 71: 253, 76: 269, 80: 476, 85: 285, 89: 174, 94: 291, 98: 219, 3: 48, 6: 119, 9: 150, 15: 161, 18: 160, 28: 249, 34: 244, 47: 251, 56: 179, 59: 200, 69: 214, 72: 291, 78: 266, 81: 242, 88: 366, 97: 270, 2: 27, 5: 118, 11: 157, 19: 176, 27: 206, 30: 252, 36: 216, 39: 155, 55: 257, 61: 269, 64: 308, 86: 292, 92: 292, 14: 189, 24: 169, 43: 237, 48: 299, 63: 236, 68: 284, 73: 213, 82: 301, 32: 260, 96: 358, 7: 117, 10: 191, 23: 204, 38: 221, 46: 225, 51: 251, 74: 216, 84: 254, 77: 240, 93: 246, 60: 304, 65: 230, 90: 283, 21: 221, 42: 271, 57: 349, 52: 244, 1: 10, 99: 49}
tries = {0: 2213, 4: 243, 8: 303, 12: 321, 16: 349, 20: 368, 25: 396, 29: 291, 33: 328, 37: 276, 41: 333, 45: 297, 50: 460, 54: 303, 58: 303, 62: 311, 66: 440, 70: 356, 75: 399, 79: 97, 83: 312, 87: 278, 91: 276, 95: 263, 13: 303, 17: 316, 22: 303, 26: 300, 31: 236, 35: 320, 40: 423, 44: 302, 49: 143, 53: 315, 67: 223, 71: 299, 76: 323, 80: 559, 85: 336, 89: 206, 94: 349, 98: 253, 3: 125, 6: 286, 9: 285, 15: 292, 18: 260, 28: 350, 34: 320, 47: 327, 56: 221, 59: 248, 69: 252, 72: 353, 78: 310, 81: 288, 88: 422, 97: 313, 2: 62, 5: 344, 11: 311, 19: 286, 27: 305, 30: 353, 36: 313, 39: 210, 55: 315, 61: 330, 64: 361, 86: 329, 92: 330, 14: 326, 24: 262, 43: 302, 48: 379, 63: 292, 68: 346, 73: 258, 82: 342, 32: 351, 96: 401, 7: 274, 10: 358, 23: 311, 38: 302, 46: 295, 51: 314, 74: 257, 84: 311, 77: 283, 93: 288, 60: 360, 65: 277, 90: 319, 21: 348, 42: 351, 57: 434, 52: 294, 1: 40, 99: 66}
plot_dump_results(corrects, tries)

