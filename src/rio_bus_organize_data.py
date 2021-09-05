from lib.data_visualization import *
from lib.rio_data_filter import *
import pandas as pd

data_path_all = '/Volumes/SD Leo/bus_gps_data/treatedGPSDataWithoutEmptyLinesSeparateColumns.csv'
data_path = '/Volumes/SD Leo/bus_gps_data/treatedGPSDataOnlyRoute.csv'

print('initiating...')

print('reading data4...')

df = pd.read_csv('/Volumes/SD Leo/bus_gps_data/organizedData4.csv')

print(df)
print('data read!')

print('droping speed column...')

#df = df.drop(columns=['speed'])

#df.to_csv('/Volumes/SD Leo/bus_gps_data/organizedData1.csv', index=False)

print('creating datetime...')

#df['datetime'] = pd.to_datetime(df.date + 'T' + df.time)

#df.to_csv('/Volumes/SD Leo/bus_gps_data/organizedData2.csv', index=False)

print('droping other columns...')

#df = df.drop(columns=['date', 'time'])

#df.to_csv('/Volumes/SD Leo/bus_gps_data/organizedData3.csv', index=False)

print('removing .0...')

#df.line = df.line.apply(lambda x: x.split('.0')[0])

#df.to_csv('/Volumes/SD Leo/bus_gps_data/organizedData4.csv', index=False)

print('sorting...')

df = df.sort_values(by= ['line', 'order', 'datetime'])

print('saving...')

df.to_csv('/Volumes/SD Leo/bus_gps_data/organizedData.csv', index=False)
