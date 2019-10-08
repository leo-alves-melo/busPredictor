#Encoding: UTF-8
import sys
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages")
import pandas as pd
from lib.classes import *
from lib.data_filter import *
from lib.data_processor import *
from lib.training_path import *
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from joblib import dump, load
import json

path_list = [2914, 3554, 3810, 3938, 4162, 4322, 4738, 4834, 5218, 5282, 5314, 5410, 5538, 5698, 5762, 5794, 5890, 6018, 6562, 6594, 7202, 7266, 7298, 7362, 7426, 7458, 7554, 7682, 7778, 8002, 8226, 8322, 8354, 8386, 8418, 8450, 8482, 8514, 8578, 8610, 8642, 8674, 8706, 8738, 8770, 8802, 8834, 8866, 8898, 8930, 8994, 9058, 9090, 9122, 9218, 9250, 9314, 9410, 9442, 9474, 9570, 9634, 9666, 9762, 9922, 10018, 10050, 10114, 10146, 10178, 10274, 10306, 10754, 10978, 11138, 11234, 11746, 12034, 12226, 12322, 12578, 12834, 12994, 13026, 13090, 13186, 13282, 13314, 13378, 13506, 13666, 13762, 14018, 14178, 14370, 14658, 14690]
paths_df = pd.read_csv('../data/media_paths_date.csv')

train_df = pd.read_csv('../data/train_df.csv', header=None)
y_df = train_df[180]
X_df = train_df.drop(columns=[180])

X_train_all, X_test, y_train_all, y_test = train_test_split(X_df, y_df, random_state=1, test_size=0.2, stratify=y_df)

X_train_all = X_train_all.abs()
X_test = X_test.abs()

model_bayes = ComplementNB().fit(X_train_all, y_train_all)


ids = {'id1': 0, 'id2': 0, 'id3': 0, 'id4': 0}

for path_id in path_list:
	current_path_df = paths_df[paths_df.index_path == path_id]

	print('path_id: ' + str(path_id))

	current_path_df = paths_df[paths_df.index_path == path_id]

	line = current_path_df.iloc[0].id_line

	#save_path(current_path_df, '../images/path_' + str(path_id) + '.png')
	current_train_df = pd.DataFrame([create_training_path(current_path_df.head(len(current_path_df.index)))])
	predicted_bayes = model_bayes.predict(current_train_df)
	print(predicted_bayes)

print ids