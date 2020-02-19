import arrow
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
from lib import *
import paho.mqtt.client as mqtt
import json
import time
from datetime import datetime, timedelta

def date_now():
    return (arrow.utcnow().to('America/Sao_Paulo') - timedelta(minutes=0)).isoformat()

base_api = 'https://api.demo.konkerlabs.net'
username = "student_smartcampus@konkerlabs.com"
password = 'SdExnFiHcGrK'
application = 'default'
channel = 'info'
circulino = "3b7c728c-cf13-45ca-a803-d94a598113d0"
circulino1 = "0b525e25-6962-4ee4-8f8a-834e3e33698d"
circulino2 = "8ab6ae21-9d9e-4637-8c56-60ba15691fd2"
circulino3 = "8f4ab4d2-203e-4043-b4b1-bffca39f8686"
circulino4 = "6ce968a1-a32b-4f8c-bc39-4463f50f4591"
circulino5 = "a028f083-8375-45ec-a42f-7e864fd5f8d7"
circulino6 = "eb0ec27f-16d0-4504-a331-af5b1e24eac6"
circulino7 = "826f5fe4-ec9b-4de7-87f8-93be40963612"
circulino8 = "8a33a2fe-8325-4c71-b937-8cc75b85bca4"

guid = circulino7

client = BackendApplicationClient(client_id=username)
oauth = OAuth2Session(client=client)
token = oauth.fetch_token(token_url='{}/v1/oauth/token'.format(base_api),client_id=username,client_secret=password)
initial_timestamp = date_now()
print(initial_timestamp)
while True:
    time.sleep(5)
    print("Tentando...")
    stats = oauth.get("https://api.demo.konkerlabs.net/v1/{}/incomingEvents?q=channel:{} timestamp:>{}&sort=latest&limit=100".format(application, channel, initial_timestamp)).json()['result']
    if stats != None and len(stats) > 0:
        initial_timestamp = date_now()
        print(json.loads(stats))
    
    