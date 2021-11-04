#Encoding: UTF-8
from keras.models import Sequential
import keras
from keras.utils import to_categorical
from keras.layers import Dense, Conv1D, Flatten, Reshape, MaxPooling1D, GlobalAveragePooling1D, Dropout, BatchNormalization
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump, load

best_nn_location = '../data/rio_best_conv_nn'

paths_df = pd.read_csv('../data/rio_train_df_60_labeled.csv', header=None)

y_df = paths_df[60*3]
X_df = paths_df.drop(columns=[60*3])

X_train, X_cross, y_train, y_cross = train_test_split(X_df, y_df, random_state=1, test_size=0.2, stratify=y_df)

y_train = to_categorical(y_train.apply(lambda x: x-1))
y_cross = to_categorical(y_cross.apply(lambda x: x-1))

height = 60
width = 3

callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='acc', patience=1)
]

possible_filters = [30, 100, 300]
possible_kernels = [3, 10]
possible_dropout = [0.3, 0.5, 0.8]

filter_1_count = 100
kernel_1_size = 10
filter_2_count = 100
kernel_2_size = 10
pooling_height = 3
filter_3_count = 160
kernel_3_count = 10
filter_4_count = 160
kernel_4_count = 5
dropout = 0.5

best_filter = 0
best_kernel = 0
best_drop = 0
best_acc = 0
best_model = None

# Find the best value of kernel, filter and dropout
for filter_count in possible_filters:
    for kernel_size in possible_kernels:
        for dropout in possible_dropout:

            filter_1_count = filter_count
            filter_2_count = filter_count
            filter_3_count = filter_count

            kernel_1_size = kernel_size
            kernel_2_size = kernel_size
            kernel_3_size = kernel_size
            
            dropout = dropout

            print('Testing for filter: ' + str(filter_count) + ' - kernel: ' + str(kernel_size) + ' - drop: ' + str(dropout))

            model = Sequential()
            model.add(Reshape((height, width), input_shape=(height*width,)))
            
            model.add(Conv1D(filter_1_count, kernel_1_size, activation='relu', input_shape=(height, width)))
            # Add batch normalization
            model.add(BatchNormalization())
            
            model.add(Conv1D(filter_2_count, kernel_2_size, activation='relu'))
            model.add(MaxPooling1D(pooling_height))
            model.add(Conv1D(filter_3_count, kernel_3_count, activation='relu'))
            model.add(Conv1D(filter_4_count, kernel_4_count, activation='relu'))
            model.add(GlobalAveragePooling1D())
            model.add(Dropout(dropout))
            model.add(Dense(19, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            model.fit(X_train, y_train, batch_size=500, epochs=50, callbacks=callbacks_list, validation_split=0.2, verbose=1)
            acc = model.evaluate(X_cross, y_cross)[1]
            if acc > best_acc:
                best_acc = acc
                best_filter = filter_count
                best_kernel = kernel_size
                best_drop = dropout
                best_model = model
                print('New bests: filter' + str(best_filter) + ' - kernel: ' + str(best_kernel) + ' - drop: ' + str(best_drop)+ ' - acc: ' + str(best_acc))
                model.save(best_nn_location)