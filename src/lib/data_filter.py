# Data Filter Functions
import datetime
import pandas as pd
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
import pprint
import json
import pandas as pd
import numpy as np
#import arrow
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
#from functools import reduce
from math import sin, cos, sqrt, atan2, radians
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.cm as cm
from datetime import timedelta
import pandas as pd

from constants import *
from classes import *

# Calculates the distance in km
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

# Calculates the distance in km
def distance_between_literal(lat1, lon1, lat2, lon2):
	# approximate radius of earth in km
	R = 6373.0

	lat1 = radians(lat1)
	lon1 = radians(lon1)
	lat2 = radians(lat2)
	lon2 = radians(lon2)

	dlon = lon2 - lon1
	dlat = lat2 - lat1
	
	a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
	c = 2 * atan2(sqrt(a), sqrt(1 - a))
	
	distance = R * c
	
	return distance

def collect_time(timestamp):
	
	if isinstance(timestamp, str):
		try:
			return datetime.datetime.strptime(timestamp, time_formatter1)
		except:
			try:
				return datetime.datetime.strptime(timestamp, time_formatter2)
			except: 
				return datetime.datetime.strptime(timestamp, time_formatter3)
	elif isinstance(timestamp, pd.Series):
		return timestamp[0]
	else:
		if isinstance(timestamp, unicode):
			return collect_time(timestamp.encode('ascii','ignore'))
		else:
			return timestamp

# Calculates the velocity in km/h
def velocity_between(position1, position2):
	
	distance = distance_between(position1, position2)
	
	time1 = collect_time(position1.timestamp)
		
	time2 = collect_time(position2.timestamp)
	
	time_delta = (time2 - time1).total_seconds()/3600
	
	if time_delta == 0:
		return 0

	return distance/time_delta

def has_distance_from_home(path_df, distance = 1):
	return (path_df.apply(lambda row: distance_between(Coordinate(row['latitude'], row['longitude']), home) < distance, axis=1) == True).any()

def has_distance_greater_from_home(path_df, distance = 1):
	return (path_df.apply(lambda row: distance_between(Coordinate(row['latitude'], row['longitude']), home) > distance, axis=1) == True).any()

def has_distance_from_coordinate(path_df, coordinate, distance = 1):
	class Coordinate():
		def __init__(self, latitude, longitude, timestamp = None, line_id = None, guid = None):
			self.latitude = latitude
			self.longitude = longitude
			self.timestamp = timestamp
			self.line_id = line_id
			self.device_guid = guid
	return (path_df.apply(lambda row: distance_between(Coordinate(row['latitude'], row['longitude']), coordinate) < distance, axis=1) == True).any()

def parse_coordinate(json):
	try:
		return Coordinate(json['geolocation']['lat'], 
						  json['geolocation']['lon'], 
						  json['timestamp'], 
						  json['payload']['id_linha'], 
						  json['incoming']['deviceGuid'])
	except:
		return Coordinate(json['geolocation']['lat'], 
						  json['geolocation']['lon'])

# Append the next coordenate if it is valid. Returns false when the path ended
def validate_and_append_next_coordinate(row, path_df, has_distance_from_greater_from_home): 
	
	coordinate = Coordinate(row['latitude'], row['longitude'], row['timestamp'], row['line_id'])
	
	# Checks if it is the first coordinate of the path
	if len(path_df.index) == 0:
		
		# Check if is close to the home, and if is, append the coordinate
		if distance_between(coordinate, home) < 0.1:
			return True, pd.DataFrame().append(row), has_distance_from_greater_from_home
		else:
			return True, pd.DataFrame(), has_distance_from_greater_from_home
		
	last_row = path_df.tail(1)
	last_coordinate = Coordinate(last_row['latitude'].iloc[0], last_row['longitude'].iloc[0], last_row['timestamp'].iloc[0], last_row['line_id'].iloc[0])
	
	# Check if is duplicated
	if coordinate == last_coordinate:
		return True, path_df, has_distance_from_greater_from_home
	
	# Checks if the velocity of the next coordinate is too big or greater than zero
	if velocity_between(coordinate, last_coordinate) > 100 and velocity_between(coordinate, last_coordinate) > 0:
		return True, path_df, has_distance_from_greater_from_home
	
	# Checks the time difference

	time1 = collect_time(last_coordinate.timestamp)
	time2 = collect_time(coordinate.timestamp)
	  
	if (time2 - time1).total_seconds() > 5*60:
		return False, path_df, has_distance_from_greater_from_home
	
	# Checks the id
	if coordinate.line_id != last_coordinate.line_id:
		return False, path_df, has_distance_from_greater_from_home
	
	# Check if the path have at least one coordinate far away from the home, so that the path started
	if has_distance_from_greater_from_home:
		# If we are close to home
		if distance_between(coordinate, home) < 0.1:
			new_df = path_df.append(row)
			return False, new_df, has_distance_from_greater_from_home
		else:
			return True, path_df.append(row), has_distance_from_greater_from_home
	else:
		if distance_between(coordinate, home) > 1:
			return True, path_df.append(row), True
		else:
			return True, path_df.append(row), False
	
def create_paths(df):

	paths_df = pd.DataFrame()
	current_path_df = pd.DataFrame()
	path_index = 0
	count_int = 0
	total = len(df.index)
	print('iniciando...')
	has_distance_from_greater_from_home = False
	for index, row in df.iterrows():
		
		if 100*index/total > count_int + 1:
			count_int += 1
			#print str(count_int) + "% - " + str(len(paths_df.index)),
			if count_int % 10 == 0:
				print('')
			
			
		still_appending, current_path_df, has_distance_from_greater_from_home = validate_and_append_next_coordinate(row, current_path_df, has_distance_from_greater_from_home)
		if not still_appending:
			
			# If there is few values on the path, we can discart it
			if len(current_path_df.index) < 100:
				current_path_df = pd.DataFrame()
				
			else:
				
				if has_distance_from_greater_from_home:
					
					path_index += 1
					# Add a column with the index of the path to the df
					current_path_df['index_path'] = path_index
					# Add the new path to the dataframe of paths
					paths_df = paths_df.append(current_path_df)
					current_path_df = pd.DataFrame()
					
				else:
					current_path_df = pd.DataFrame()
			
			has_distance_from_greater_from_home = False

						
	return paths_df