# Data Visualization

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
#from constants import *

def plot_path_with_map(path_df, map_path, coordinates_box):
	img = plt.imread(map_path)
	fig, ax = plt.subplots()
	latitudes = path_df.latitude
	longitudes = path_df.longitude
	numberOfDots = len(path_df.index)
	colors = cm.rainbow(np.linspace(0, 1, numberOfDots))

	ax.scatter(longitudes, latitudes, color=colors)

	ax.set_xlim(coordinates_box[0], coordinates_box[1])
	ax.set_ylim(coordinates_box[2], coordinates_box[3])

	ax.imshow(img, zorder=0, extent=coordinates_box, aspect= 'equal')

	plt.show()
	plt.clf()

def plot_path(path_df):

	latitudes = path_df.latitude
	longitudes = path_df.longitude
	
	numberOfDots = len(path_df.index)
	colors = cm.rainbow(np.linspace(0, 1, numberOfDots))
	
	plt.margins(0) # Fit the map in the image
	
	plt.scatter(longitudes, latitudes, color=colors)
	
	#plt.scatter(bus_stop_longitude, bus_stop_latitude, marker='*')
		
	plt.show()

def save_path(path_df, name):

	latitudes = path_df.latitude
	longitudes = path_df.longitude
	
	numberOfDots = len(path_df.index)
	colors = cm.rainbow(np.linspace(0, 1, numberOfDots))
	
	plt.margins(0) # Fit the map in the image
	
	plt.scatter(longitudes, latitudes, color=colors)
	
	#plt.scatter(bus_stop_longitude, bus_stop_latitude, marker='*')
		
	plt.savefig(name)
	plt.clf()

def plot_correctness(correctness):
	plt.plot(range(len(correctness)), correctness)
	plt.ylabel('Porcentagem')
	plt.xlabel('# Leituras')
	plt.title('Porcentagem de acerto por # de leituras')
	plt.grid(True)
	plt.show()