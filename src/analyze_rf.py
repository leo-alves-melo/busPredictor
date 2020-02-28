import pandas as pd
from joblib import dump, load

from lib.classes import *
from lib.constants import *
from lib.data_filter import *
from lib.data_normalization import *
from lib.data_processor import *
from lib.training_path import *

data_path = "data/"
best_rf_location = data_path + 'best_rf.joblib'
media_path_location = data_path + "media_paths_date.csv"

model_rf = load(best_rf_location)

paths_df = pd.read_csv(media_path_location)

minimum_path = int(paths_df.index_path.min())
maximum_path = int(paths_df.index_path.max()) + 1
total = paths_df.index_path.max() + 1
count_int = 0

max_lenght = 0

deslocamento = 1

correctness = {}

for lenght in range(101):
    correctness["total_1_" + str(lenght)] = 0
    correctness["total_2_" + str(lenght)] = 0
    correctness["total_3_" + str(lenght)] = 0
    correctness["total_4_" + str(lenght)] = 0

    correctness["correct_rf_1_" + str(lenght)] = 0
    correctness["correct_rf_2_" + str(lenght)] = 0
    correctness["correct_rf_3_" + str(lenght)] = 0
    correctness["correct_rf_4_" + str(lenght)] = 0

    correctness["error_rf_1_" + str(lenght)] = 0
    correctness["error_rf_2_" + str(lenght)] = 0
    correctness["error_rf_3_" + str(lenght)] = 0
    correctness["error_rf_4_" + str(lenght)] = 0

for path_id in range(minimum_path + deslocamento, maximum_path, 32):

    print('path_id: ' + str(path_id))

    current_path_df = paths_df[paths_df.index_path == path_id]

    line = int(current_path_df.iloc[0].id_line)
    
    for lenght in range(1, len(current_path_df.index), 2):

        if lenght > max_lenght:
            max_lenght = lenght

        current_path_df_lenght = current_path_df.head(lenght)
        current_train_df = pd.DataFrame([create_training_path(current_path_df_lenght)])
        predicted_rf = int(model_rf.predict(current_train_df)[0])

        lenght = int(100*lenght/len(current_path_df.index) + 1)

        # Append the new number of total of this lenght and add if it was correct
        correctness['total_' + str(line) + "_" + str(lenght)] += 1

        if predicted_rf == line:
            correctness['correct_rf_' + str(line) + "_"  + str(lenght)] += 1
        else:
            correctness['error_rf_' + str(predicted_rf) + "_"  + str(lenght)] += 1

print(correctness)