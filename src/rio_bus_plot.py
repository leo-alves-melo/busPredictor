from lib.data_visualization import *
from lib.rio_data_filter import *
import pandas as pd

data_path_all = '/Volumes/SD Leo/bus_gps_data/treatedGPSDataWithoutEmptyLinesSeparateColumns.csv'
data_path = '/Volumes/SD Leo/bus_gps_data/treatedGPSDataOnlyRoute.csv'
small_riobus = '../data/mini_riobus.csv'

df = pd.read_csv(small_riobus, parse_dates=['datetime'], dtype={'line': object})

file = open('../data/riobus_coordinates.json')
data = json.load(file)

final_stops = {}
for key in data.keys():
    final_stops[key] = [Coordinate(latitude=data[key][0][0], longitude=data[key][0][1]), Coordinate(latitude=data[key][1][0], longitude=data[key][1][1])]

df = df[df.line.isin(data.keys())]

path = create_paths(df, final_stops)
print(path)

exit()
for index in range(len(small_path.index)):
    if not has_started:
        if distance_between(small_path.iloc[index], final_stops[371][0]) < minimun_distance or distance_between(small_path.iloc[index], final_stops[371][1]) < minimun_distance:
            has_started = True
    #print(has_started)
    print(str(small_path.iloc[index].latitude) + ', ' + str(small_path.iloc[index].longitude))
    #print(distance_between(small_path.iloc[index], final_stops[371][0]), distance_between(small_path.iloc[index], final_stops[371][1]))

#plot_path(small_path)

