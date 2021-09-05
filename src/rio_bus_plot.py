from lib.data_visualization import *
from lib.rio_data_filter import *
import pandas as pd

data_path_all = '/Volumes/SD Leo/bus_gps_data/treatedGPSDataWithoutEmptyLinesSeparateColumns.csv'
data_path = '/Volumes/SD Leo/bus_gps_data/treatedGPSDataOnlyRoute.csv'

df = pd.read_csv(data_path_all)

line = '371.0'
final_stops = {line: [Coordinate(latitude=-22.907849, longitude=-43.184382), Coordinate(latitude=-22.901384, longitude=-43.345651)]}

small_path = df[df.order == 'C51623'][df.line == line]

small_path.to_csv('small.csv')
exit()
paths = []
lines = [371]
minimun_distance = 0.8

path = create_paths(small_path, final_stops)
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

