# Importing packages

from oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
import pprint
import json
import pandas as pd
import numpy as np
import arrow
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
from functools import reduce
from math import sin, cos, sqrt, atan2, radians
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.cm as cm

# Setting variables

base_api = 'https://api.demo.konkerlabs.net'
username = "student_smartcampus@konkerlabs.com"
password = 'SdExnFiHcGrK'

circulino = "3b7c728c-cf13-45ca-a803-d94a598113d0"
circulino1 = "0b525e25-6962-4ee4-8f8a-834e3e33698d"
circulino2 = "8ab6ae21-9d9e-4637-8c56-60ba15691fd2"
circulino3 = "8f4ab4d2-203e-4043-b4b1-bffca39f8686"
circulino4 = "6ce968a1-a32b-4f8c-bc39-4463f50f4591"
circulino5 = "a028f083-8375-45ec-a42f-7e864fd5f8d7"
circulino6 = "eb0ec27f-16d0-4504-a331-af5b1e24eac6"
circulino7 = "826f5fe4-ec9b-4de7-87f8-93be40963612"
circulino8 = "8a33a2fe-8325-4c71-b937-8cc75b85bca4"

# These are the available circulars because the others do not have data
available_circulars = [circulino4, circulino5, circulino6, circulino7, circulino8]

# Coordinates of the home of the circular
latitude_home = -22.829636
longitude_home = -47.061038
home = {'geolocation':{'lat': latitude_home, 'lon': longitude_home}}

time_formatter1 = '%Y-%m-%dT%H:%M:%S.%fZ'
time_formatter2 = '%Y-%m-%dT%H:%M:%SZ'

bus_stop_latitude = [-22.82961372593675,-22.827572007972478,-22.827839460784613,-22.82572604482161,-22.824259395268815,-22.824342054788538,-22.82339495615896,-22.823696505309023,-22.819732610271284,-22.818471307010142,-22.818031511340962,-22.818063107264045,-22.81897827715408,-22.81851763367064,-22.816911924943703,-22.8149687910257,-22.81348716556454,-22.812519511209437,-22.812994160952606,-22.81423669359395,-22.81654917375286,-22.819294695494193,-22.82167567621637,-22.825048660966733,-22.827383217435795,-22.82880780070171,-22.830572518710206,-22.82966004880576,-22.82961372593675,-22.827572007972478,-22.827839460784613,-22.82572604482161,-22.824234047789606,-22.822700953548658,-22.821469396127068,-22.819146975421376,-22.816566655760123,-22.81417771866976,-22.813895379025784,-22.813831568436914,-22.812777378217994,-22.81348716556454,-22.81504658684353,-22.81561825362484,-22.817031676329528,-22.819334903296923,-22.821000894424408,-22.821312063645497,-22.819994834014597,-22.817413656323698,-22.81471355050243,-22.81566370714975,-22.818063107264045,-22.81826851842413,-22.818182857637566,-22.81834806053481,-22.819694150746503,-22.823807510190303,-22.82486161502456,-22.827383217435795,-22.82880780070171,-22.830572518710206,-22.82966004880576,-22.82961372593675,-22.827572007972478,-22.827839460784613,-22.82572604482161,-22.824234047789606,-22.822700953548658,-22.821469396127068,-22.819146975421376,-22.816566655760123,-22.81417771866976,-22.813895379025784,-22.813831568436914,-22.812777378217994,-22.81348716556454,-22.81504658684353,-22.81561825362484,-22.817031676329528,-22.819334903296923,-22.821000894424408,-22.821312063645497,-22.819994834014597,-22.817413656323698,-22.81471355050243,-22.81466547432292,-22.81404623926024,-22.81348716556454,-22.81566370714975,-22.818063107264045,-22.81826851842413,-22.818182857637566,-22.81834806053481,-22.819694150746503,-22.823807510190303,-22.82486161502456,-22.827383217435795,-22.82880780070171,-22.830572518710206,-22.82966004880576,-22.82961372593675,-22.827572007972478,-22.827839460784613,-22.82572604482161,-22.824234047789606,-22.822700953548658,-22.821469396127068,-22.819146975421376,-22.816566655760123,-22.81417771866976,-22.813895379025784,-22.814990643788022,-22.817031676329528,-22.819334903296923,-22.821000894424408,-22.821312063645497,-22.819994834014597,-22.817413656323698,-22.81471355050243,-22.81566370714975,-22.818063107264045,-22.824468293279722,-22.82486161502456,-22.827383217435795,-22.82880780070171,-22.830572518710206,-22.82966004880576]
bus_stop_longitude = [-47.06137016415596,-47.06275820732117,-47.06677682697773,-47.06618271768093,-47.06345960497856,-47.059839963912964,-47.060167863965034,-47.061488181352615,-47.06003777682781,-47.05960728228092,-47.0604857057333,-47.06308342516422,-47.064886540174484,-47.06610158085823,-47.06609085202217,-47.06696391105652,-47.065243273973465,-47.06756003201008,-47.06950530409813,-47.07185357809061,-47.07285739481449,-47.07241617143154,-47.070336788892746,-47.067298516631126,-47.067575454711914,-47.066226303577366,-47.06210911273956,-47.06069424748421,-47.06137016415596,-47.06275820732117,-47.06677682697773,-47.06618271768093,-47.06772968173027,-47.06905469298363,-47.07022547721863,-47.07222707569599,-47.07261733710766,-47.071487456560135,-47.069668248295784,-47.06821247935295,-47.0677524805069,-47.065243273973465,-47.06436015665531,-47.06585146486759,-47.06623703241348,-47.06711947917938,-47.068121284246445,-47.067411839962006,-47.06532709300518,-47.063652724027634,-47.064024209976196,-47.06265024840826,-47.06308342516422,-47.063074707984924,-47.06050112843508,-47.05967500805855,-47.06013835966587,-47.06294797360897,-47.06547796726227,-47.067575454711914,-47.066226303577366,-47.06210911273956,-47.06069424748421,-47.06137016415596,-47.06275820732117,-47.06677682697773,-47.06618271768093,-47.06772968173027,-47.06905469298363,-47.07022547721863,-47.07222707569599,-47.07261733710766,-47.071487456560135,-47.069668248295784,-47.06821247935295,-47.0677524805069,-47.065243273973465,-47.06436015665531,-47.06585146486759,-47.06623703241348,-47.06711947917938,-47.068121284246445,-47.067411839962006,-47.06532709300518,-47.063652724027634,-47.064024209976196,-47.05708935856819,-47.05889314413065,-47.065243273973465,-47.06265024840826,-47.06308342516422,-47.063074707984924,-47.06050112843508,-47.05967500805855,-47.06013835966587,-47.06294797360897,-47.06547796726227,-47.067575454711914,-47.066226303577366,-47.06210911273956,-47.06069424748421,-47.06137016415596,-47.06275820732117,-47.06677682697773,-47.06618271768093,-47.06772968173027,-47.06905469298363,-47.07022547721863,-47.07222707569599,-47.07261733710766,-47.071487456560135,-47.069668248295784,-47.067154347896576,-47.06623703241348,-47.06711947917938,-47.068121284246445,-47.067411839962006,-47.06532709300518,-47.063652724027634,-47.064024209976196,-47.06265024840826,-47.06308342516422,-47.06490330398077,-47.06547796726227,-47.067575454711914,-47.066226303577366,-47.06210911273956,-47.06069424748421]

# Data Collector Functions

def get_data(guid):
    global oauth
    
    dt_start = arrow.utcnow().to('America/Sao_Paulo').floor('day')
    timestamp_init = dt_start.shift(days=-7).isoformat()
    channel = 'info'
    timestamp_final = arrow.utcnow().isoformat()
    application = 'default'
    result = []
    lastlen = 0
    while timestamp_init < timestamp_final:
        try:
            stats = oauth.get("https://api.demo.konkerlabs.net/v1/{}/incomingEvents?q=device:{} channel:{} timestamp:>{}&sort=oldest&limit=10000".format(application,guid, channel, timestamp_init)).json()['result']
            if (len(stats)<2) and (lastlen<20000):
                break
            timestamp_init = stats[-1]['timestamp']
            result.extend(stats)
            lastlen = len(stats)
        except:
            pass
    return result

# Data Filter Functions

# Calculates the distance in km
def distance_between(position1, position2):
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(position1['geolocation']['lat'])
    lon1 = radians(position1['geolocation']['lon'])
    lat2 = radians(position2['geolocation']['lat'])
    lon2 = radians(position2['geolocation']['lon'])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    distance = R * c
    
    return distance

# Calculates the velocity in km/h
def velocity_between(position1, position2):
    
    distance = distance_between(position1, position2)
    
    try:
        time1 = datetime.strptime(position1['timestamp'], time_formatter1)
    except:
        time1 = datetime.strptime(position1['timestamp'], time_formatter2)
        
    try:
        time2 = datetime.strptime(position2['timestamp'], time_formatter1)
    except:
        time2 = datetime.strptime(position2['timestamp'], time_formatter2)
    time_delta = (time2 - time1).total_seconds()/3600
    
    if time_delta == 0:
        return 0

    return distance/time_delta

# Append the next coordenate if it is valid. Returns false when the path ended
def validate_and_append_next_coordinate(coordinate, path): 
    
    # Checks if it is the first coordinate of the path
    if len(path) == 0:
        
        # Check if is close to the home, and if is, append the coordinate
        if distance_between(coordinate, home) < 0.1:
            return True, [coordinate]
        else:
            return True, []
    
    # Check if is duplicated
    if coordinate == path[-1]:
        return True, path
    
    # Checks if the velocity of the next coordinate is too big or greater than zero
    if velocity_between(coordinate, path[-1]) > 100 and velocity_between(coordinate, path[-1]) > 0:
        return True, path
    
    # Checks the time difference
    try:
        time1 = datetime.strptime(path[-1]['timestamp'], time_formatter1)
    except:
        time1 = datetime.strptime(path[-1]['timestamp'], time_formatter2)
        
    try:
        time2 = datetime.strptime(coordinate['timestamp'], time_formatter1)
    except:
        time2 = datetime.strptime(coordinate['timestamp'], time_formatter2)
        
    if (time2 - time1).total_seconds() > 5*60:
        return False, path
    
    # Checks the id
    if coordinate['payload']['id_linha'] != path[-1]['payload']['id_linha']:
        return False, path
    
    # Check if the path have at least one coordinate far away from the home, so that the path started
    biggest_distance = 0
    for path_coordinate in path:
        if distance_between(path_coordinate, home) > biggest_distance:
            biggest_distance = distance_between(path_coordinate, home)
    
    # If the path already started
    if biggest_distance > 1:
        # If we are close to home
        if distance_between(coordinate, home) < 0.1:
            return False, path + [coordinate]
        else:
            return True, path + [coordinate]
    else:
        return True, path + [coordinate]
    
def create_paths(data):
    paths = [[]]
    index = 0
    count = 0
    count_int = 0
    total = len(data)
    for coordinate in data:
        count += 1
        if 100*count/total > count_int + 1:
            count_int += 1

            print(str(count_int) + "%")
        still_appending, paths[index] = validate_and_append_next_coordinate(coordinate, paths[index])
        if not still_appending:
            # If there is few values on the path, we can discart it
            if len(paths[index]) < 105:
                paths[index] = []
            else:
                index += 1
                paths.append([])
    
    return paths

# Data Visualization

def plot_path(path):

    latitudes = [coordinate['geolocation']['lat'] for coordinate in path]
    longitudes = [coordinate['geolocation']['lon'] for coordinate in path]
    
    numberOfDots = len(path)
    colors = cm.rainbow(np.linspace(0, 1, numberOfDots))
    
    plt.margins(0) # Fit the map in the image
    
    plt.scatter(latitudes, longitudes, color=colors)
    
    plt.scatter(bus_stop_latitude, bus_stop_longitude, marker='*')
        
    plt.show()

# Data Processing Functions

def average_window(window):
    latitude_sum = reduce(lambda summation, coordinate: summation + coordinate['geolocation']['lat'], window, 0)
    longitude_sum = reduce(lambda summation, coordinate: summation + coordinate['geolocation']['lon'], window, 0)
    
    latitude_avg = latitude_sum/len(window)
    longitude_avg = longitude_sum/len(window)
    
    coordinate_avg = window[0].copy()
    coordinate_avg['geolocation']['lat'] = latitude_avg
    coordinate_avg['geolocation']['lon'] = longitude_avg
    
    return coordinate_avg
    
def apply_average_window_on_path(path, window_size = 10, window_shift = 2):
    index = 0
    new_path = []
    while index < len(path):
        window = path[index:index + window_size]
        new_coordinate = average_window(window)
        new_path.append(new_coordinate)
        index += window_shift
        
    return new_path

# Authenticating

client = BackendApplicationClient(client_id=username)
oauth = OAuth2Session(client=client)
token = oauth.fetch_token(token_url='{}/v1/oauth/token'.format(base_api),client_id=username,client_secret=password)

# Reading all circular data

data = []

for circular in available_circulars:
    data += get_data(circular)

paths = create_paths(data)

for i in range(5):
    print('iteration: ' + str(i) + ' - id: ' + str(paths[i][0]['payload']['id_linha']))
    plot_path(paths[i])
    