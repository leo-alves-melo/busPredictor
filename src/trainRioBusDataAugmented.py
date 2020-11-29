from keras.models import Sequential
import keras
from keras.utils import to_categorical
from keras.layers import Dense, Conv1D, Conv2D, Flatten, Reshape, MaxPooling1D, MaxPooling2D, GlobalAveragePooling1D, Dropout, BatchNormalization
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump, load
import numpy as np

# Normalizing X considering the absolute values of the coordinates
highest_latitude = 22.836006
lowest_latitude = 22.801396
highest_longitude = 47.095658
lowest_longitude = 47.046078
hightest_time = 86400
lowest_time = 0
def adjustTimeColumn(time):
	if time == 0:
		return 0
	return (time - lowest_time)/(hightest_time - lowest_time)

def adjustLatitudeColumn(latitude):
	if latitude == 0:
		return 0
	return (latitude - lowest_latitude)/(highest_latitude - lowest_latitude)

def adjustLongitudeColumn(longitude):
	if longitude == 0:
		return 0
	return (longitude - lowest_longitude)/(highest_longitude - lowest_longitude)

def adjustTrainDf(df):

	for column in df:
		# For time columns
		if column % 3 == 0:
			df[column] = df[column].apply(adjustTimeColumn)
		elif column % 3 == 1:
			df[column] = df[column].apply(adjustLatitudeColumn)
		elif column % 3 == 2:
			df[column] = df[column].apply(adjustLongitudeColumn)

	return df

def convertTo3D(df):
    time_train_df = []
    lat_train_df = []
    lon_train_df = []
    for line in range(len(df.index)):

        time_train_df.append(list(df.iloc[line][df.iloc[line].index % 3 == 0].to_numpy()))
        lat_train_df.append(list(df.iloc[line][df.iloc[line].index % 3 == 1].to_numpy()))
        lon_train_df.append(list(df.iloc[line][df.iloc[line].index % 3 == 2].to_numpy()))
    new_df = np.array([time_train_df, lat_train_df, lon_train_df])
    return new_df.reshape((new_df.shape[1], 60, 3))

# Constants and global variables

train_location = '../data/train_df.csv'
best_nn_location = '../data/best_nn_cross_rio_bus_data.h5'

train_df = []

print("lendo dados...")

for index in range(2000):
    df = pd.read_csv(filename, index=False)
    train_df.append(df)

print("Criando df...")

train_df = pd.concat(train_df, axis=0, ignore_index=True)

print("dados lidos!")
print(train_df)

y_df = train_df[180]
X_df = train_df.drop(columns=[180])

X_train_all, X_test, y_train_all, y_test = train_test_split(X_df, y_df, random_state=1, test_size=0.2, stratify=y_df)

#X_train_all = adjustTrainDf(X_train_all.abs())

X_train, X_cross, y_train, y_cross = train_test_split(X_train_all, y_train_all, random_state=1, test_size=0.2, stratify=y_train_all)

#X_test = adjustTrainDf(X_test.abs())

y_train = to_categorical(y_train.apply(lambda x: x-1))
y_cross = to_categorical(y_cross.apply(lambda x: x-1))
y_test = to_categorical(y_test.apply(lambda x: x-1))

X_train = convertTo3D(X_train)
X_cross = convertTo3D(X_cross)

print("treinando...")

model = Sequential()

model.add(Conv1D(64, 3, activation='relu', input_shape=(60,3)))

model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(128, 3, activation='relu'))

model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())

model.add(Dense(2000, activation='sigmoid'))
model.add(Dense(2000, activation='sigmoid'))
model.add(Dense(4, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, batch_size=500, epochs=50, validation_split=0.2, verbose=1)
acc = model.evaluate(X_cross, y_cross)[1]

print('acc: ' + str(acc))

model.save(best_nn_location)