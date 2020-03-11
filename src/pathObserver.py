import arrow
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
from lib.classes import *
from lib.constants import *
from lib.data_filter import *
from lib.data_normalization import *
from lib.data_processor import *
from lib.training_path import *
import paho.mqtt.client as mqtt
import json
import time
from datetime import datetime, timedelta
import pandas as pd
from joblib import dump, load

data_path = "data/"
best_rf_location = data_path + 'best_rf.joblib'
model_rf = load(best_rf_location)

# Length of each line id
line_id_lengths = {1: 8100, 2: 10000, 3: 12000, 4: 8610}
line_tp_percentage = {1: 0.15, 2: 0.85, 3: 0, 4: 0.38}

def sort_stats(stat):
    return collect_time(stat['timestamp'])

def date_now():
    return (arrow.utcnow().to('America/Sao_Paulo') - timedelta(minutes=1)).isoformat()

def convert_to_seconds(text):
    if isinstance(text, float):
        return text
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
        return True, df, has_distance_from_greater_from_home

def check_paths(paths):
    global model_rf

    for guid in paths.keys():
        if len(paths[guid]["path"].index) > 0:
            path = paths[guid]["path"].copy()
            path['timestamp'] = path['timestamp'].apply(lambda x: convert_to_seconds(x))
            path = path.rename(columns={"timestamp": "date"})
            print(path)
            line = int(path.iloc[0].line_id)
            training_path = pd.DataFrame([create_training_path(path)])
            predicted_rf = int(model_rf.predict(training_path)[0])

            print("Comparando...")
            print("Deu: " + str(predicted_rf) + " - e era pra dar: " + str(line))
    
            length = path_length(path)
            print("percorrido: " + str(length))

            completed_percentual_line = 100 * length / line_id_lengths[line]
            completed_percentual_predicted = 100 * length / line_id_lengths[predicted_rf]

            print("Percentual line: " + str(completed_percentual_line) + "%")
            print("Percentual predicted: " + str(completed_percentual_predicted) + "%")

            if predicted_rf != line:
                if line_tp_percentage[line] < completed_percentual_line:
                    print("Acho que a linha esta incorreta")

                    if line_tp_percentage[line] < completed_percentual_predicted:
                        print("Acho que a linha correta eh: " + str(predicted_rf))

def path_length(path):
    length = 0
    print(len(path.index))
    for index in range(1, len(path.index)):
        coordinate1 = Coordinate(path.iloc[index-1].latitude, path.iloc[index-1].longitude)
        print(coordinate1.latitude)
        coordinate2 = Coordinate(path.iloc[index].latitude, path.iloc[index].longitude)
        length += distance_between(coordinate1, coordinate2)
    
    return length

base_api = 'https://api.demo.konkerlabs.net'
username = "student_smartcampus@konkerlabs.com"
password = 'SdExnFiHcGrK'
application = 'default'
channel = 'info'

client = BackendApplicationClient(client_id=username)
oauth = OAuth2Session(client=client)
token = oauth.fetch_token(token_url='{}/v1/oauth/token'.format(base_api),client_id=username,client_secret=password)
initial_timestamp = date_now()

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

            if paths[guid]["still_appending"] == False:
                paths[guid]["path"] = pd.DataFrame()
                paths[guid]["has_distance_from_greater_from_home"] = False
        #print(paths)
    
        check_paths(paths)
    time.sleep(5)
    
    