import arrow
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
from lib.classes import *
from lib.constants import *
from lib.data_filter import *
from lib.data_normalization import *
from lib.data_processor import *
import paho.mqtt.client as mqtt
import json
import time
from datetime import datetime, timedelta
import pandas as pd

def sort_stats(stat):
    return collect_time(stat['timestamp'])

def date_now():
    return (arrow.utcnow().to('America/Sao_Paulo') - timedelta(minutes=1)).isoformat()

def convert_to_seconds(text):
    date = collect_time(text)
    return (date - date.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()

def create_or_append_path(path, location, has_distance_from_greater_from_home):
    
    if path is None:
        df = pd.DataFrame()
    else:
        df = path
    try:
        row = pd.io.json.json_normalize([{'timestamp' : collect_time(location['timestamp']), 
                'latitude': location['geolocation']['lat'], 
                'longitude': location['geolocation']['lon'],
                'line_id': location['payload']['id_linha']}])

        return validate_and_append_next_coordinate(row, df, has_distance_from_greater_from_home)
    except:
        print("Opa deu ruim")
        print(location)
        return True, df, has_distance_from_greater_from_home

base_api = 'https://api.demo.konkerlabs.net'
username = "student_smartcampus@konkerlabs.com"
password = 'SdExnFiHcGrK'
application = 'default'
channel = 'info'

client = BackendApplicationClient(client_id=username)
oauth = OAuth2Session(client=client)
token = oauth.fetch_token(token_url='{}/v1/oauth/token'.format(base_api),client_id=username,client_secret=password)
initial_timestamp = date_now()
print(initial_timestamp)
paths = {}
while True:
    print("Tentando...")
    stats = oauth.get("https://api.demo.konkerlabs.net/v1/{}/incomingEvents?q=channel:{} timestamp:>{}&sort=latest&limit=100".format(application, channel, initial_timestamp)).json()['result']
    if stats != None and len(stats) > 0:
        initial_timestamp = date_now()
        stats.sort(key=sort_stats)
        for location in stats:
            guid = location["incoming"]["deviceGuid"]
            has_distance_from_greater_from_home = False
            if not (paths.get(guid) is None):
                if paths[guid].get("has_distance_from_greater_from_home") == True:
                    has_distance_from_greater_from_home = True
            else:
                paths[guid] = {}
            paths[guid]["still_appending"], paths[guid]["path"], paths[guid]["has_distance_from_greater_from_home"] = create_or_append_path(paths[guid].get("path"), location, has_distance_from_greater_from_home)
        print(paths)
    
    time.sleep(5)
    
    