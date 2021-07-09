import os
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import four_to_six
from tensorflow import keras
from tensorflow.keras import layers
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignore unuseful massages from tensorflow

# set train data
x_train_raw = np.loadtxt("train_data.txt", delimiter=",")  # load 65 train signals
SNR = 100  # Signal-to-noise Ratio = 100 dB - no noise should be added for training.
# (the lower the SNR value the bigger distortions in signal)
x_train = four_to_six(x_train_raw, SNR)  # preprocess 4 received signals to 6 compensation signals
# see function description in "data_preprocessing.py" and README file

# categorical output:
y_train = np.zeros((65, 65))  # correct outputs for all 65 signals
np.fill_diagonal(y_train, 1)  # every signal is an individual class
# unfortunately we have a very little amount of data because of
# computation difficulties with obtaining these time dependencies
# thus in this project we have no validation dataset

x_train = np.transpose(x_train)
y_train = np.transpose(y_train)

# Create neural network
model = keras.Sequential(
    [
        layers.InputLayer(4614),
        layers.Dense(1024, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(65, activation='softmax')
    ]
)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

his = model.fit(x_train, y_train, epochs=100, batch_size=5)

plt.plot(his.history['loss'])
plt.grid()
plt.show()

# # uncomment to save other networks:
# model.save('ANN_2_valid/')
