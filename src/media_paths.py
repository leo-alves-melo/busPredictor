import pandas as pd
from lib.classes import *
from lib.data_filter import *
from lib.data_processor import *
import datetime

paths_df = pd.read_csv('../data/paths.csv')
paths_df = paths_df.rename(columns={'Path Index': 'index_path'})
paths_df = paths_df.drop(columns=['Unnamed: 0'])

paths_df['date'] = pd.to_datetime(paths_df['date'])

paths_df.date = paths_df.date.apply(lambda row: row.hour*3600 + row.minute*60 +row.second)

total = paths_df.index_path.max() + 1
count_int = 0

avg_df = pd.DataFrame()

for path_id in range(paths_df.index_path.min(), paths_df.index_path.max() + 1):
	
	if 100 * path_id / total > count_int:
		count_int += 1
		print count_int
		avg_df.to_csv('../data/media_paths_date.csv', index=False)

	current_path_df = paths_df[paths_df.index_path == path_id]
	
	current_avg_df = apply_average_time_window(current_path_df)

	avg_df = avg_df.append(current_avg_df)

avg_df.to_csv('../data/media_paths_date.csv', index=False)
