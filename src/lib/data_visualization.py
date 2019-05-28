# Data Visualization

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from constants import *

def plot_path(path_df):

	latitudes = path_df.latitude
	longitudes = path_df.longitude
	
	numberOfDots = len(path_df.index)
	colors = cm.rainbow(np.linspace(0, 1, numberOfDots))
	
	plt.margins(0) # Fit the map in the image
	
	plt.scatter(longitudes, latitudes, color=colors)
	
	plt.scatter(bus_stop_longitude, bus_stop_latitude, marker='*')
		
	plt.show()

def plot_correctness(correctness):
	plt.plot(range(len(correctness)), correctness)
	plt.ylabel('Porcentagem')
	plt.xlabel('# Leituras')
	plt.title('Porcentagem de acerto por # de leituras')
	plt.grid(True)
	plt.show()