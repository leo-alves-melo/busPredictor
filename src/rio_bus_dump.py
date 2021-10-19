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

def test_dump_all_possibilities(df, model):

	# test will be indexed in 3, 7, 11...

	lines = df.line.unique()
	jump = 64

	correct = {}
	total_try = {}

	errors = {}
	super_errors = {}

	for line in lines:
		print('line:', line)
		current_df = df[df.line == line]

		errors[line] = {}
		super_errors[line] = {}

		possible_index_paths = current_df.index_path.unique()
		total = possible_index_paths[-1] - possible_index_paths[0]
		count = 0
		for index_path in range(possible_index_paths[3], possible_index_paths[-1], jump):
			
			count += (1*jump)
			current_path = current_df[current_df.index_path == index_path]
			
			path_lenght = len(current_path.index)

			predictions = model.predict_all_possibilities(current_path)

			if line in predictions:
				for prediction in predictions:
					if prediction != line:
						if errors[line].get(prediction) == None:
							errors[line][prediction] = [index_path]
						else:
							errors[line][prediction].append(index_path)
			else:
				for prediction in predictions:
					if super_errors[line].get(prediction) == None:
						super_errors[line][prediction] = [index_path]
					else:
						super_errors[line][prediction].append(index_path)
				

			print(count/total *100, '- errors:', errors, '- super_errors:', super_errors)
	
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

	def predict_all_possibilities(self, path):
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
		
		current_possible_lines = self.possible_lines

		self.possible_lines = self.paths.line.unique()
		return current_possible_lines

def plot_dump_results(corrects, tries):
	result = []
	
	for index in range(4, 100, 5):
		corrects_sum = corrects[index] + corrects[index - 1] + corrects[index - 2] + corrects[index - 3] + corrects[index - 4]
		total_sum = tries[index] + tries[index - 1] + tries[index - 2] + tries[index - 3] + tries[index - 4]
		result.append(100*corrects_sum/total_sum)

	result.append(88.6) # This result was computed with completed paths
	plt.plot(list(range(0, 101, 5)), result, label=u'Georreferência Automático', marker='', markerfacecolor='blue')
	
	plt.xlabel('Porcentagem de Completude')
	plt.ylabel(u'Acurácia')
	plt.title(u'Porcentagem de Completude X Acurácia do Trajeto')
	plt.legend(loc='best')
	plt.grid(True)

	plt.show()

def create_errors_table(errors):
	df = pd.DataFrame(columns=['keys'] + list(errors.keys()))
	for key in errors.keys():
		row = errors[key]
		
		for inside_keys in row.keys():
			row[inside_keys] = [len(row[inside_keys])]
		row['keys'] = [key]

		row = pd.DataFrame(row)
		df = df.append(row)
	df = df.fillna(0)
	return df

def testErrors(df, errors, box_coordinates):
	humans = 0
	others = 0
	for line in errors.keys():
		print('line:', line, '- humans:', humans, '- others:', others)
		for inline in errors[line].keys():
			indexes = errors[line][inline]
			print('--inline:', inline)
			for index in indexes:
				current_df = df[df.index_path == index]
				print('-- -- foi configurado', line, 'mas foi predito', inline)
				plot_path_with_map(current_df, '../data/mapa_rio.png', box_coordinates)
				if input('tipo de erro:') == 'h':
					humans += 1
				else:
					others += 1
	print('humans:', humans, '- others:', others)
	




small_riobus = '../data/rio_paths_10.csv'
minimum_percent = 0.1
minimum_distance = 0.1

#models_df = pd.read_csv('../data/dump_models.csv', parse_dates=['datetime'], dtype={'line': object})
df = pd.read_csv(small_riobus, parse_dates=['datetime'], dtype={'line': object})

#model = DumpRioBusAlgorithm(models_df)

#test_dump_all_possibilities(df, model)

min_lon = -43.4168
max_lon = -42.9173
min_lat = -23.0440
max_lat = -22.7799

box_coordinates = (min_lon, max_lon, min_lat, max_lat)

corrects_dump = {0: 452, 4: 90, 8: 139, 12: 176, 16: 193, 20: 247, 25: 276, 29: 199, 33: 233, 37: 207, 41: 263, 45: 230, 50: 371, 54: 248, 58: 250, 62: 255, 66: 367, 70: 299, 75: 342, 79: 77, 83: 267, 87: 228, 91: 225, 95: 229, 13: 182, 17: 191, 22: 195, 26: 209, 31: 163, 35: 241, 40: 319, 44: 238, 49: 112, 53: 252, 67: 185, 71: 253, 76: 269, 80: 476, 85: 285, 89: 174, 94: 291, 98: 219, 3: 48, 6: 119, 9: 150, 15: 161, 18: 160, 28: 249, 34: 244, 47: 251, 56: 179, 59: 200, 69: 214, 72: 291, 78: 266, 81: 242, 88: 366, 97: 270, 2: 27, 5: 118, 11: 157, 19: 176, 27: 206, 30: 252, 36: 216, 39: 155, 55: 257, 61: 269, 64: 308, 86: 292, 92: 292, 14: 189, 24: 169, 43: 237, 48: 299, 63: 236, 68: 284, 73: 213, 82: 301, 32: 260, 96: 358, 7: 117, 10: 191, 23: 204, 38: 221, 46: 225, 51: 251, 74: 216, 84: 254, 77: 240, 93: 246, 60: 304, 65: 230, 90: 283, 21: 221, 42: 271, 57: 349, 52: 244, 1: 10, 99: 49}
tries_dump = {0: 2213, 4: 243, 8: 303, 12: 321, 16: 349, 20: 368, 25: 396, 29: 291, 33: 328, 37: 276, 41: 333, 45: 297, 50: 460, 54: 303, 58: 303, 62: 311, 66: 440, 70: 356, 75: 399, 79: 97, 83: 312, 87: 278, 91: 276, 95: 263, 13: 303, 17: 316, 22: 303, 26: 300, 31: 236, 35: 320, 40: 423, 44: 302, 49: 143, 53: 315, 67: 223, 71: 299, 76: 323, 80: 559, 85: 336, 89: 206, 94: 349, 98: 253, 3: 125, 6: 286, 9: 285, 15: 292, 18: 260, 28: 350, 34: 320, 47: 327, 56: 221, 59: 248, 69: 252, 72: 353, 78: 310, 81: 288, 88: 422, 97: 313, 2: 62, 5: 344, 11: 311, 19: 286, 27: 305, 30: 353, 36: 313, 39: 210, 55: 315, 61: 330, 64: 361, 86: 329, 92: 330, 14: 326, 24: 262, 43: 302, 48: 379, 63: 292, 68: 346, 73: 258, 82: 342, 32: 351, 96: 401, 7: 274, 10: 358, 23: 311, 38: 302, 46: 295, 51: 314, 74: 257, 84: 311, 77: 283, 93: 288, 60: 360, 65: 277, 90: 319, 21: 348, 42: 351, 57: 434, 52: 294, 1: 40, 99: 66}

errors = {'100': {'108': [1476, 7108, 16196, 17988], '104': [1476, 16900, 22340], '109': [2244, 8516, 24004], '2251': [16772, 24388]}, '101': {}, '102': {'104': [38172], '100': [38172]}, '104': {'109': [38307, 39523, 40227, 40355, 40483, 40611, 41059, 41763, 41891, 42275, 42467, 42915, 43427, 43683, 44579, 44771, 44835, 45027, 45155, 45283, 46115, 46307, 46371, 46627, 47331, 48227, 49059, 49507, 50083], '108': [38371, 38435, 39907, 40547, 42339, 42659, 42787, 43043, 43171, 43939, 44515, 45475, 45603, 45795, 47011, 47779, 47907, 48483, 48611, 48803, 48995, 49123, 49443], '133': [42659, 43043, 43939, 44643, 45603, 45731, 47779, 47907, 48803, 49891], '102': [49891]}, '108': {'210': [61670], '104': [61670], '209': [61670]}, '109': {'104': [63527, 64039, 65191]}, '11': {}, '117': {'108': [66403, 67939, 70819, 71011, 72163, 77091]}, '118': {}, '133': {}, '209': {'133': [86679, 87127, 87255, 87447, 88023, 88087, 88215, 88407, 88535, 88855, 88983, 89111, 89495, 89687, 89815, 90071, 90391, 90519, 91159, 91607, 91735, 92375, 92695]}, '210': {'100': [92943, 93071, 93199, 93903, 94415, 94607, 94863, 95183, 95375, 95503, 96911, 97167], '222': [92943, 93071, 94863, 95375], '209': [93519, 95311, 95631, 95887, 96463]}, '2110': {}, '2111': {'2110': [99851]}, '2112': {}, '2114': {'2110': [106021, 106149, 108773, 108901, 109413, 109477, 110949, 111269], '2111': [108453, 109413, 109477, 110309, 110565, 111653], '2112': [111397]}, '2115': {}, '220': {}, '222': {}, '2251': {'108': [139901], '100': [139901], '220': [139901]}}
super_errors = {'100': {'2112': [2052], '104': [3012, 3268, 16644, 24836], '108': [6084, 24772], '2115': [14788], '2114': [19908], '102': [21316, 25604], '101': [24068], '2251': [27332]}, '101': {'102': [28686, 28814], '108': [30478]}, '102': {'108': [31772, 33628, 35548, 36700], '104': [32924, 36252, 37532], '101': [36636], '209': [38044]}, '104': {'102': [38627, 39715], '108': [39843, 40163, 41507, 43107, 47651], '109': [40739, 41827, 49955], '101': [42083], '210': [42531, 48547, 49635], '100': [42723, 47587], '133': [43747, 46883], '2114': [45091]}, '108': {'100': [50214, 55910, 61798], '104': [50342, 52838, 53414, 53926, 54566, 54694, 55654, 57062, 57382, 57958, 59302, 59942, 61030, 61222], '133': [50854], '102': [55078, 59110, 62118]}, '109': {'104': [62951, 63399, 63463, 63847, 64359, 65447], '100': [63079, 63143, 63655, 63719, 64935], '108': [65063, 65127, 65319]}, '11': {'100': [65624]}, '117': {'109': [65635, 74339], '100': [65635, 66595, 68387, 69539, 73443, 75171, 76003, 76067, 76515, 76643], '2111': [74851], '2112': [74851], '2114': [74851], '2110': [74851], '222': [74851]}, '118': {'117': [77145]}, '133': {'104': [78728, 80648, 84424, 85128], '108': [80840], '109': [82568], '101': [84424, 86088], '209': [86024], '100': [86216]}, '209': {'210': [90327, 91095, 91991], '133': [91671], '104': [91927, 92503], '2112': [92247]}, '210': {'209': [92879, 93583, 94159, 95695, 96015, 96271, 96335, 96399, 97103], '100': [93135, 94095, 97039], '222': [93135]}, '2110': {'2114': [98274, 98466, 98914, 99042, 99362, 99426, 99618], '2111': [98530, 98594, 98722]}, '2111': {'2114': [100043, 100363, 100811, 101259, 101835, 102219, 102411], '2110': [100875, 101067, 101643, 102283], '2112': [101387, 102027]}, '2112': {'100': [104286, 104990], '104': [104670], '2114': [105054, 105118, 105502], '2111': [105118]}, '2114': {'2112': [105893, 106277, 108965, 109861, 110117, 111077], '2111': [106469, 107685, 108709, 109093], '2110': [106533, 106725, 106917, 106981, 108261, 108709, 109733, 111589], '2251': [107237]}, '2115': {'2114': [111833, 111897, 111961]}, '220': {'104': [124543], '133': [126783, 130623]}, '222': {'2112': [134030, 134158], '11': [134222]}, '2251': {'2111': [135997], '2114': [135997, 139581], '2110': [135997], '100': [136125, 136189, 137661, 139581], '2112': [137789]}}

quantities = {46: 1658, 77: 1076, 43: 1709, 96: 448, 39: 1928, 87: 733, 56: 1678, 89: 669, 68: 1505, 64: 1674, 33: 2075, 78: 1215, 54: 1609, 73: 1301, 82: 907, 176: 33, 51: 1651, 69: 1524, 66: 1573, 74: 1275, 28: 2240, 84: 818, 67: 1558, 63: 1627, 52: 1601, 88: 696, 30: 2118, 102: 294, 70: 1468, 40: 1865, 48: 1680, 42: 1754, 92: 516, 83: 878, 55: 1711, 37: 1940, 80: 1061, 72: 1309, 53: 1704, 29: 2176, 45: 1670, 41: 1865, 296: 10, 62: 1621, 57: 1632, 90: 639, 95: 477, 27: 2169, 44: 1698, 50: 1672, 99: 357, 140: 72, 202: 23, 26: 2089, 180: 37, 34: 2084, 59: 1645, 76: 1182, 38: 2026, 35: 2021, 86: 757, 65: 1626, 36: 2068, 60: 1721, 58: 1608, 49: 1679, 61: 1713, 104: 249, 143: 46, 127: 93, 25: 2100, 313: 13, 79: 1096, 47: 1642, 31: 2198, 75: 1204, 97: 431, 118: 142, 145: 63, 24: 1917, 115: 155, 85: 789, 71: 1387, 23: 1997, 32: 2140, 114: 190, 289: 14, 91: 607, 81: 945, 170: 30, 199: 21, 19: 1785, 101: 341, 147: 68, 20: 1834, 131: 73, 132: 72, 154: 40, 105: 301, 126: 73, 161: 29, 93: 495, 153: 51, 624: 1, 157: 28, 165: 39, 122: 110, 142: 40, 21: 1866, 111: 175, 156: 40, 434: 8, 245: 11, 499: 1, 158: 46, 110: 220, 106: 256, 144: 52, 124: 88, 155: 43, 22: 1954, 15: 1458, 133: 75, 139: 55, 203: 23, 251: 20, 112: 196, 267: 11, 173: 22, 193: 18, 94: 470, 178: 27, 125: 104, 98: 393, 757: 1, 160: 36, 501: 2, 214: 23, 187: 26, 225: 21, 138: 55, 119: 118, 108: 205, 103: 319, 141: 57, 120: 117, 100: 381, 135: 70, 223: 19, 168: 36, 231: 23, 247: 12, 171: 36, 1158: 1, 191: 29, 159: 33, 384: 6, 12: 1073, 221: 24, 162: 36, 227: 17, 150: 46, 248: 18, 265: 7, 196: 26, 152: 53, 109: 220, 107: 250, 236: 16, 184: 31, 175: 32, 116: 147, 189: 22, 123: 101, 113: 177, 380: 5, 181: 31, 297: 11, 194: 23, 473: 2, 174: 30, 14: 1203, 166: 38, 18: 1745, 11: 1057, 182: 20, 13: 1146, 169: 28, 149: 46, 17: 1668, 198: 21, 834: 1, 117: 156, 167: 41, 222: 19, 343: 10, 250: 11, 192: 21, 419: 10, 550: 2, 121: 104, 148: 53, 163: 40, 151: 49, 134: 78, 349: 8, 446: 4, 506: 2, 164: 37, 136: 68, 200: 17, 146: 45, 424: 2, 219: 14, 186: 21, 314: 11, 323: 8, 137: 78, 130: 76, 1301: 1, 442: 5, 293: 11, 351: 4, 217: 18, 209: 24, 16: 1423, 10: 958, 279: 6, 129: 94, 362: 8, 242: 9, 275: 15, 266: 15, 246: 16, 492: 4, 185: 31, 268: 11, 179: 21, 177: 25, 207: 22, 257: 17, 407: 4, 383: 5, 353: 6, 399: 5, 363: 14, 337: 7, 220: 12, 367: 5, 188: 30, 197: 22, 128: 86, 274: 17, 243: 18, 340: 9, 201: 17, 1048: 1, 205: 18, 172: 31, 339: 5, 241: 12, 400: 7, 190: 22, 264: 13, 204: 14, 405: 6, 256: 12, 249: 12, 372: 3, 368: 3, 233: 14, 213: 21, 252: 10, 445: 3, 481: 3, 408: 3, 306: 11, 417: 5, 208: 16, 318: 7, 210: 24, 438: 5, 195: 20, 312: 2, 273: 9, 212: 23, 239: 17, 295: 5, 228: 9, 235: 15, 423: 3, 333: 6, 294: 6, 281: 12, 307: 9, 211: 22, 287: 9, 371: 6, 303: 9, 278: 10, 269: 12, 352: 3, 300: 15, 387: 6, 255: 9, 345: 4, 412: 5, 356: 3, 240: 16, 230: 15, 224: 23, 494: 2, 385: 7, 288: 4, 409: 2, 390: 5, 855: 1, 308: 13, 254: 13, 428: 1, 373: 3, 183: 24, 276: 6, 344: 10, 347: 8, 237: 19, 284: 13, 272: 9, 234: 9, 465: 2, 280: 7, 450: 2, 259: 13, 229: 17, 216: 19, 299: 11, 325: 11, 2000: 1, 335: 9, 1762: 1, 1033: 2, 1671: 1, 292: 12, 271: 16, 283: 9, 395: 4, 215: 16, 311: 8, 497: 3, 261: 12, 238: 15, 253: 20, 441: 3, 260: 11, 258: 12, 291: 11, 206: 15, 317: 8, 381: 4, 262: 9, 1401: 1, 244: 13, 290: 9, 298: 9, 302: 9, 629: 1, 429: 4, 370: 5, 538: 3, 786: 2, 316: 9, 226: 15, 310: 6, 366: 4, 357: 8, 470: 3, 369: 4, 585: 1, 377: 3, 361: 6, 218: 11, 355: 3, 394: 1, 540: 2, 498: 1, 603: 2, 505: 2, 487: 2, 474: 2, 901: 1, 651: 2, 232: 18, 713: 1, 595: 1, 342: 12, 599: 2, 341: 7, 413: 2, 558: 3, 382: 6, 336: 6, 304: 12, 322: 5, 965: 1, 1516: 1, 410: 4, 398: 4, 375: 2, 560: 1, 543: 2, 397: 2, 453: 2, 3077: 1, 958: 1, 721: 1, 328: 11, 309: 7, 510: 1, 484: 2, 565: 3, 360: 1, 329: 6, 477: 2, 513: 1, 523: 3, 1408: 1, 541: 2, 270: 8, 427: 3, 467: 2, 354: 2, 263: 10, 365: 4, 697: 1, 326: 5, 458: 2, 548: 2, 657: 1, 813: 1, 479: 2, 338: 4, 462: 3, 301: 11, 415: 4, 430: 3, 402: 2, 332: 6, 406: 5, 528: 1, 2309: 1, 282: 5, 572: 2, 305: 6, 763: 1, 623: 1, 616: 2, 431: 1, 391: 4, 286: 8, 787: 1, 471: 2, 708: 1, 472: 1, 646: 1, 285: 9, 330: 6, 491: 2, 331: 5, 374: 9, 277: 8, 774: 1, 1029: 2, 486: 2, 334: 8, 359: 2, 327: 6, 452: 4, 324: 10, 350: 3, 388: 7, 556: 2, 319: 6, 348: 7, 574: 1, 386: 5, 1109: 1, 451: 4, 425: 3, 950: 1, 443: 2, 389: 3, 421: 2, 4029: 1, 392: 2, 346: 6, 1125: 1, 511: 3, 755: 1, 459: 1, 426: 4, 1573: 1, 433: 1, 502: 1, 466: 2, 404: 3, 321: 3, 411: 1, 422: 3, 449: 2, 461: 2, 1140: 1, 439: 2, 379: 3, 436: 4, 828: 1, 396: 6, 378: 2, 440: 1, 364: 3, 393: 4, 500: 3, 488: 2, 1629: 1, 315: 3, 1132: 1, 1950: 1, 877: 1, 483: 1, 2292: 1, 525: 1, 515: 1, 530: 1, 1022: 1, 539: 2, 715: 1, 564: 1, 861: 1, 418: 1, 3324: 1, 808: 1, 685: 1, 696: 1, 600: 1, 1846: 1, 358: 3, 469: 2, 569: 1, 601: 3, 432: 2, 743: 1, 1108: 1, 1320: 1, 793: 1, 435: 2, 414: 2, 1344: 1, 673: 1, 447: 1, 320: 2, 437: 2, 482: 3, 514: 2, 401: 2, 597: 1, 1338: 1, 1002: 1, 537: 1, 547: 2, 519: 2, 1501: 1, 468: 1, 1693: 1, 655: 2, 756: 1, 493: 1, 576: 1, 613: 1, 794: 1, 420: 1, 622: 1, 480: 2, 5339: 1, 448: 1, 778: 1, 566: 1, 3369: 1, 577: 1, 588: 1, 7123: 1, 507: 1, 533: 1, 444: 1, 508: 1, 476: 1, 1047: 1, 457: 2, 542: 1, 621: 1, 551: 2, 3328: 1, 7197: 1, 608: 1, 711: 1, 512: 1, 1212: 1, 3640: 1, 518: 2, 531: 1, 1392: 1, 570: 1, 455: 1, 1081: 1, 521: 1, 496: 1}

testErrors(df, super_errors, box_coordinates)

#errors_table = create_errors_table(errors)
#super_errors_table = create_errors_table(super_errors)

#errors_table.to_csv('../data/rio_errors_table.csv', index=False)
#super_errors_table.to_csv('../data/rio_super_errors_table.csv', index=False)

