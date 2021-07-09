import os
import numpy as np
from data_preprocessing import four_to_six, set_index, visualize
from tensorflow import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignore unuseful massages from tensorflow

model = keras.models.load_model('ANN_1/')  # load neural network
test_data = np.loadtxt("train_data.txt", delimiter=",")  # load train data
# in this project we will test the ANN on the noised train data

obj_and_pos = 'PMN-4_25cm'  # here you can set which object on which distance you want to test
# it is important to follow the list of combination of objects and distances in "recognize()" function:
# objects = ['can1', 'can2', 'can3', 'can4', 'can5', 'can6', 'PMN-1', 'PMN-4']
# distances = ['_0cm', '_5cm', '_10cm', '_15cm', '_20cm', '_25cm', '_30cm', '_35cm']
test_sig_index = set_index(obj_and_pos)  # returns the index of signal of desired object and position
SNR = 30  # Signal-to-Noise Ratio. The lower the ratio, the bigger the distortions
M = 10  # number of iterations. Every SNR realization is quite unique because of random nature
# We need to obtain a result depending not on the realization sample but on the
# power of noise. Thus the best way is to make a big number of realizations to exclude
# a random impact for the answer. The bigger number of realizations (M) - the lower random impact is.
# bet there is a trade-off between accuracy of result and calculation speed.
# It is recommended to use at least 500 realizations for reliable result.

main_answer = np.zeros((65, M))  # zero array for ANN answers
for i in range(0, M):
    test_signal = np.array(test_data[:, test_sig_index], ndmin=2)  # get a chosen signal from array
    test_signal_1 = four_to_six(np.transpose(test_signal), SNR)  # noise it and form 6 stitched signals

    answer = model.predict(np.transpose(test_signal_1))  # make a prediction by ANN
    answer = answer.flatten()  # answer must be a 1D raw for using in further loop

    # here we set a response of maximum neuron to 1, and nullify non-maximum neurons.
    # a statical answer will be easier to get by this way
    for j in range(0, len(answer)):
        if answer[j] != max(answer):
            answer[j] = 0
        else:
            answer[j] = 1

    main_answer[:, i] = answer[:]  # save the result vector in zero array

visualize(main_answer, obj_and_pos, SNR)  # visualization of answer
