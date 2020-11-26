import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def plot_path(path_df):

	latitudes = path_df.latitude
	longitudes = path_df.longitude
	
	numberOfDots = len(path_df.index)
	colors = cm.rainbow(np.linspace(0, 1, numberOfDots))
	
	plt.scatter(longitudes, latitudes, color=colors)
	
	#plt.scatter(bus_stop_longitude, bus_stop_latitude, marker='*')
		
	plt.show()

media_path_location = '../data/media_paths_date.csv'

train_location = '../data/'

paths_df = pd.read_csv(media_path_location)
new_train_df = paths_df[paths_df.index_path % 4 == 0].copy()
model_train_df = new_train_df.copy()

multiply = 2000

current_index_path = paths_df.index_path.max() + 1

for mult in range(1, multiply + 1):

	current_df = model_train_df.copy()
	size = len(current_df.index)

	current_df.date += np.random.normal(0, 60, size)
	current_df.latitude += np.random.normal(0, 0.00005, size)
	current_df.longitude += np.random.normal(0, 0.0001, size)

	current_df.index_path += (mult*current_df.index_path.max())

	new_train_df = new_train_df.append(current_df)

new_train_df.to_csv('../data/augmented_media_paths_date.csv', index=False)