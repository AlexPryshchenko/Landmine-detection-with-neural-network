import numpy as np
import matplotlib.pyplot as plt


def add_noise(sig, snr):
    # this functions adds white Gaussian noise to the time dependence
    # sig - input signal
    # snr - signal-to-noise ratio

    sig_wats = sig ** 2  # Raw signals are in V/m
    target_snr_db = snr  # Set a target SNR

    sig_avg_watts = np.mean(sig_wats)  # Calculate signal power and convert to dB
    sig_avg_db = 10 * np.log10(sig_avg_watts)

    noise_avg_db = sig_avg_db - target_snr_db  # Calculate noise and convert to watts
    noise_avg_watts = 10 ** (noise_avg_db / 10)

    mean_noise = 0  # Generate an sample of white noise
    noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(sig_wats))

    sig_with_noise = sig + noise_volts  # Noise up the original signal
    # The lower the ratio, the bigger the distortions!
    return sig_with_noise


def four_to_six(data_inp, snr1):
    # this function creates compensation signals from 4 received signals by Ground Penetrating Radar
    # data_inp - input signal
    # snr - signal-to-noise ratio

    step4 = 769  # in every column we have 4 signals with 769 points in time.
    # (time step = 0.01 ns. Thus 769 values of electric field in 7.69 ns)
    # Because of this function will be used in other projects with other time steps of signals,
    # i've decided to pick out this parameter.

    diff_total_mass = np.zeros((step4 * 6, len(data_inp[0, :])))  # zero array.
    for i in range(0, len(data_inp[0, :])):
        # get 4 signals from column
        sig1 = data_inp[0:step4, i]
        sig2 = data_inp[step4:2 * step4, i]
        sig3 = data_inp[2 * step4:3 * step4, i]
        sig4 = data_inp[3 * step4:4 * step4, i]

        # noise them up
        sig1_n = add_noise(sig1, snr1)
        sig2_n = add_noise(sig2, snr1)
        sig3_n = add_noise(sig3, snr1)
        sig4_n = add_noise(sig4, snr1)

        # get 6 signals (see readme.md)
        diff_1 = sig1_n - sig2_n
        diff_2 = sig1_n + sig3_n
        diff_3 = sig1_n + sig4_n
        diff_4 = sig2_n + sig3_n
        diff_5 = sig2_n + sig4_n
        diff_6 = sig3_n - sig4_n

        # stitch them in 1 column
        diff_total = np.concatenate((diff_1, diff_2), axis=0)
        diff_total = np.concatenate((diff_total, diff_3), axis=0)
        diff_total = np.concatenate((diff_total, diff_4), axis=0)
        diff_total = np.concatenate((diff_total, diff_5), axis=0)
        diff_total = np.concatenate((diff_total, diff_6), axis=0)
        diff_total_mass[:, i] = diff_total  # save signals to zero array and continue loop

    return diff_total_mass


def set_index(name):
    # this function allows to set testing signal by defining the object and distance in string form
    # instead of defining a number of column of array
    # complexity of this problem is lays in fact that we cant directly see the answer just looking on
    # the received signal unlike for example in problem of digits recognition.
    # thus this function is quite necessary, just not to confuse in a numbers of columns ant its labels
    objects = ['can1', 'can2', 'can3', 'can4', 'can5', 'can6', 'PMN-1', 'PMN-4']  # define all objects
    distances = ['_0cm', '_5cm', '_10cm', '_15cm', '_20cm', '_25cm', '_30cm', '_35cm']  # define all distances
    labels = []  # zero list
    for i in range(0, 8):
        for j in range(0, 8):
            labels.append(objects[j] + distances[i])  # obtaining all cases

    test_sig = 0  # starting index
    for i in range(0, 64):
        if name == labels[i]:  # we are running through the "labels" list until defined object and
            # position (name) will not match with one of the labels[i].
            test_sig = i + 1  # then we take this index and add 1, because in this project
            # we are not going to trace 0 output of ANN, which corresponds to absence of object.
            # thus we are looking at 1-65 outputs. See the categorical decryption in READ_ME.md
    return test_sig


def visualize(main_answer, obj_and_pos, snr):
    # this function allows to make a demonstrative visualization of ANN response
    # for fixed value of SNR
    # main_answer - array of ANN answers (see ANN_testing.py)
    # obj_and_pos - testing object and position defined in ANN_testing.py
    # snr - signal-to-noise ratio

    main_answer = np.sum(main_answer, axis=1)  # here we summing up an ANN answers by raws,
    # thus obtaining a total number of hits for every output neuron.
    main_answer = main_answer[1:65]  # here we cut off first neuron, which corresponds to absence of object
    main_answer = main_answer.reshape((8, 8))  # here we create an 8x8 grid for more demonstrative visualization

    # here we are making an axis grid for 3d bar visualization of "main_answer"
    x = np.arange(0, 8, 1)
    y = x.copy()
    xpos, ypos = np.meshgrid(x, y)
    z = np.transpose(main_answer)
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)
    barsize = 0.7
    dx = barsize * np.ones_like(zpos)
    dy = dx.copy()
    dz = z.flatten()

    # on the first subplot we make an "2d" visualization with imshow
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    im = ax1.imshow(main_answer)
    ax1.set_xlabel('Objects')
    ax1.set_ylabel('Distance, cm')
    ax1.set_xticks(np.arange(0, 8, 1))
    ax1.set_yticks(np.arange(0, 8, 1))
    ax1.set_xticklabels(['can1', 'can2', 'can3', 'can4', 'can5', 'can6', 'PMN-1', 'PMN-4'])
    ax1.set_yticklabels([0, 5, 10, 15, 20, 25, 30, 35])  # our output neurons corresponds
    # exactly to this X and Y labels
    cbar = plt.colorbar(im)
    cbar.set_label('Number of realizations')  # defined by "M" in ANN_testing.py
    ax1.set_title('2D visualization')

    # on the second subplot we make an "3d" visualization with bar3d
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.bar3d(xpos, ypos, zpos, dx, dy, dz, color='blue')
    ax2.set_xlabel('Distance, cm')
    ax2.set_ylabel('Objects')
    ax2.set_zlabel('Number of realizations')
    ax2.set_xticks(np.arange(0 + barsize / 2, 8 + barsize / 2, 1))  # barsize value hepls to move ticks in
    # more representative place for almost all view angles of 3d bar
    ax2.set_xticklabels([0, 5, 10, 15, 20, 25, 30, 35])
    ax2.set_yticks(np.arange(0 + barsize / 2, 8 + barsize / 2, 1))
    ax2.set_yticklabels(['can1', 'can2', 'can3', 'can4', 'can5', 'can6', 'PMN-1', 'PMN-4'])
    ax2.set_title('3D visualization')

    fig.suptitle('ANN response for detecting a ' + obj_and_pos + ' with SNR = ' + str(snr) + ' dB')
    fig_manager = plt.get_current_fig_manager()  # full screen figure
    fig_manager.window.showMaximized()
    plt.show()
