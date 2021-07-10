import os
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import four_to_six, set_index, add_noise
from tensorflow import keras
from matplotlib.animation import ArtistAnimation

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignore unuseful massages from tensorflow

# this program creates an animation by which you can trace the ANN answer dependending on increasing the noise

model = keras.models.load_model('model1.h5')  # load neural network
test_data = np.loadtxt("train_data.txt", delimiter=",")  # load training data
# in this project we will test the ANN on the noised train data

obj_and_pos = 'PMN-1_20cm'  # here you can set which object on which distance you want to test
# it is important to follow the list of combination of objects and distances in "recognize()" function:
# objects = ['can1', 'can2', 'can3', 'can4', 'can5', 'can6', 'PMN-1', 'PMN-4']
# distances = ['_0cm', '_5cm', '_10cm', '_15cm', '_20cm', '_25cm', '_30cm', '_35cm']
test_sig_index = set_index(obj_and_pos)  # returns the index of signal of desired object and position

SNR_range = np.arange(30, 9, -1)  # a range of Signal-to-Noise Ratios which will be used in animation
# The lower the ratio, the bigger the distortions!
M = 500  # number of iterations. Every SNR realization is quite unique because of random nature
# We need to obtain a result depending not on the realization sample but on the
# power of noise. Thus the best way is to make a big number of realizations to exclude
# a random impact for the answer. The bigger number of realizations (M) - the lower random impact is.
# bet there is a trade-off between accuracy of result and calculation speed.
# It is recommended to use at least 500 realizations for reliable result.

# creating a figure
# on 3 subplots we will see an input signal distortion, 2D imshow answer and 3D view.
fig = plt.figure()
# scaling of subplots
ax = fig.add_gridspec(4, 3)
ax1 = fig.add_subplot(ax[1:3, 0])  # input signal distortion visualization
ax1.grid()

ax2 = fig.add_subplot(ax[0:, 1])  # 2D visualization
ax3 = fig.add_subplot(ax[0:, 2], projection='3d')  # 3D visualization

# here we are making an axis grid for 3d bar
x = np.arange(0, 8, 1)
y = x.copy()
xpos, ypos = np.meshgrid(x, y)
xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = np.zeros_like(xpos)
barsize = 0.7
dx = barsize * np.ones_like(zpos)
dy = dx.copy()

frames = []  # zero array of animation frames
main_answer = np.zeros((65, M))
for SNR in SNR_range:
    for i in np.arange(0, M, 1):
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

    main_answer1 = np.sum(main_answer, axis=1)  # here we summing up an ANN answers by rows,
    # thus obtaining a total number of hits for every output neuron.
    main_answer2 = main_answer1[1:65]  # here we cut off first neuron, which corresponds to absence of object
    main_answer3 = main_answer2.reshape((8, 8))  # here we create an 8x8 grid for more demonstrative visualization

    data1, = ax1.plot(add_noise(test_data[:769, test_sig_index], SNR), color="red")  # signal distortion demonstration
    ax1.plot(test_data[:769, test_sig_index], color="black")
    ax1.set_xlabel('Time, ns')
    ax1.set_xticks(np.arange(0, 800, 100))
    ax1.set_ylabel('Amplitude, V/m')
    ax1.set_xticklabels(['0', '1', '2', '3', '4', '5', '6', '7'])
    ax1.set_title('Distortion of testing signal')
    ax1.legend(['Test signal', 'Train signal'])

    data2 = ax2.imshow(main_answer3, vmin=0, vmax=M)  # 2d visualization
    ax2.set_xlabel('Objects')
    ax2.set_ylabel('Distance, cm')
    ax2.set_xticks(np.arange(0, 8, 1))
    ax2.set_yticks(np.arange(0, 8, 1))
    ax2.set_xticklabels(['can1', 'can2', 'can3', 'can4', 'can5', 'can6', 'PMN-1', 'PMN-4'], rotation=-45)
    ax2.set_yticklabels([0, 5, 10, 15, 20, 25, 30, 35])
    ax2.set_title('2D Answer')

    z = np.transpose(main_answer3)  # 3d visualization
    dz = z.flatten()
    data3 = ax3.bar3d(xpos, ypos, zpos, dx, dy, dz, color='blue')
    ax3.set_xlabel('Distance, cm')
    ax3.set_ylabel('Objects', labelpad=20)
    ax3.set_zlabel('Number of realizations')
    ax3.set_xticks(np.arange(0 + barsize / 2, 8 + barsize / 2, 1))
    ax3.set_xticklabels([0, 5, 10, 15, 20, 25, 30, 35])
    ax3.set_yticks(np.arange(0 + barsize / 2, 9 + barsize / 2, 1))
    ax3.set_yticklabels(['', 'can1', 'can2', 'can3', 'can4', 'can5', 'can6', 'PMN-1', 'PMN-4'], rotation=-45)
    ax3.set_title('3D Answer')

    figManager = plt.get_current_fig_manager()  # full-screen figure
    figManager.window.showMaximized()

    ttl = plt.figtext(0.28, 0.95,
                      'ANN response for detecting a ' + obj_and_pos + ' with SNR = ' + str(SNR) + ' dB',
                      fontsize=15)  # add an altering title

    stat = plt.figtext(0.43, 0.88,
                       str(int(main_answer1[test_sig_index])) + ' of ' + str(M) + ' realizations are correct',
                       fontsize=10)  # add an altering information about absolute number of correct answers

    frames.append([data1, data2, data3, ttl, stat])  # add frames

my_animation = ArtistAnimation(
    fig,
    frames,
    interval=250,
    blit=False,
    repeat=True)

plt.show()
