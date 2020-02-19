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

def date_now():
    return (arrow.utcnow().to('America/Sao_Paulo') - timedelta(minutes=0)).isoformat()

def create_or_append_path(path, location):
    print(location)
    df = path
    if path is None:
        df = pd.DataFrame()
    df = df.append(pd.io.json.json_normalize({'timestamp' : location['timestamp'], 
            'latitude': location['geolocation']['lat'], 
            'longitude': location['geolocation']['lon'],
            'id_line': location['payload']['id_linha']}))
    return df

base_api = 'https://api.demo.konkerlabs.net'
username = "student_smartcampus@konkerlabs.com"
password = 'SdExnFiHcGrK'
application = 'default'
channel = 'info'

guid = circulino7

client = BackendApplicationClient(client_id=username)
oauth = OAuth2Session(client=client)
token = oauth.fetch_token(token_url='{}/v1/oauth/token'.format(base_api),client_id=username,client_secret=password)
initial_timestamp = date_now()
print(initial_timestamp)
paths = {}
while True:
    time.sleep(5)
    print("Tentando...")
    stats = oauth.get("https://api.demo.konkerlabs.net/v1/{}/incomingEvents?q=channel:{} timestamp:>{}&sort=latest&limit=100".format(application, channel, initial_timestamp)).json()['result']
    if stats != None and len(stats) > 0:
        initial_timestamp = date_now()
        for location in stats:
            guid = location["incoming"]["deviceGuid"]
            paths[guid] = create_or_append_path(paths.get(guid), location)
        print(paths)
    
    