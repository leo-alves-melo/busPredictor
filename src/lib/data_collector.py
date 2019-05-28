# Data Collector Functions
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
from data_filter import *

client = BackendApplicationClient(client_id=username)
oauth = OAuth2Session(client=client)
token = oauth.fetch_token(token_url='{}/v1/oauth/token'.format(base_api),client_id=username,client_secret=password)

def get_data(guid):

	print('coletando: ' + guid)
	
	timestamp_init = datetime.datetime(2003, 9, 25, 0, 0)
	channel = 'info'
	timestamp_final = datetime.datetime.now()
	application = 'default'
	df = pd.DataFrame()
	lastlen = 0
	count = 0
	while timestamp_init < timestamp_final:

		stats = oauth.get("https://api.demo.konkerlabs.net/v1/{}/incomingEvents?q=device:{} channel:{} timestamp:>{}&sort=oldest&limit=10000".format(application,guid, channel, timestamp_init.isoformat())).json()['result']
		if (len(stats)<2) and (lastlen<20000):
			break

		timestamp_init = collect_time(stats[-1]['timestamp'])

		new_data = []
		for json in stats:
			
			try:
				new_json = {}
				new_json['timestamp'] = json.get('timestamp')
				new_json['latitude'] = json.get('geolocation').get('lat')
				new_json['longitude'] = json.get('geolocation').get('lon')
				new_json['line_id'] = json.get('payload').get('id_linha')
				new_json['device_id'] = json.get('incoming').get('deviceGuid')
				new_json['pdop'] = json.get('payload').get('_pdop')
				
				new_data.append(new_json)
			except:
				print('errado:')
				print(new_json)

		df = df.append(pd.io.json.json_normalize(new_data))
		
		lastlen = len(stats)

		count += 1
		if count % 30 == 0:
			print(timestamp_init)

	return df

def read_all_circular_data_from_server():
	# Authenticating

	client = BackendApplicationClient(client_id=username)
	oauth = OAuth2Session(client=client)
	token = oauth.fetch_token(token_url='{}/v1/oauth/token'.format(base_api),client_id=username,client_secret=password)
	
	# Reading all circular data

	df = pd.DataFrame()
	for circular in available_circulars:
		new_df = get_data(circular)
		df = df.append(new_df)
		
	return df

def download_data(data):
	with open('data.json', 'w') as file:
		file.write(json.dumps(data))

	files.download('data.json')