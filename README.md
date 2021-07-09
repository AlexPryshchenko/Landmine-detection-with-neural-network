# __Landmine detection with Neural Network and UWB radar__
This project demonstrates an artificial intellegence approach to detect and recognize hidden underground objects.

## __Requirments:__
---

* Python 3.9
* TensorFlow 2.5
* Numpy 
* Matplotlib

## __Project overview:__
---

 __ANN_1/__ (folder) - saved TF model

__ANN_training.py__ - code to create, tune and train neural network.

__ANN_testing.py__ - code to test a neural network and get a static visualization of certain answer.

__ANN_testing_with_animation.py__ - code to create animation for tracing a tendencies of neural network responce.

__train_data.txt__ - array with training data.


## __Overview of features of this approach:__
---

Ultrawideband (UWB) Ground Penetrating Radar (GPR) iradiates the model of soil, which contains a mine:

![physical problem](https://github.com/AlexPryshchenko/Landmine_detection_with_ANN/blob/main/readme_files/gif1.gif)

Reflected waves are receiving back to antenna system, and we obtain 4 time dependences:

![4 signals](https://github.com/AlexPryshchenko/Landmine_detection_with_ANN/blob/main/readme_files/fig1.png)

First peak (1) - is an reflection from a ground, second one (2) - reflection from the underground object. Second peak contains more usefull information, so we have to somehow increase its influence in time dependence. 

If you look closer to these 4 signals, you can see that 3 and 4 signals are like a mirror reflection of 1 and 2. And if you plus them - you exactly will decrese the first pick (1) and increse the second one (2) (But the total amplitude will be lower). Also there is a six unique combinations of adding and substraction of these 4 signals:

![6 signals](https://github.com/AlexPryshchenko/Landmine_detection_with_ANN/blob/main/readme_files/fig2.png)

We stitch them together and this data will serve as an input data for neural network.

> __train_data.txt__ contains signals like in first figure. 
__"four_to_six"__ function from __data_preprocessing.py__ makes a transformation for signals from form in figure 1 to form in figure 2. 

---

### __Considered underground objects:__

In this project we have 2 mines under consideration. First is PMN-1 mine:

![pmn 1](https://github.com/AlexPryshchenko/Landmine_detection_with_ANN/blob/main/readme_files/PMN-1.png)

Second is PMN-4 mine:

![pmn 4](https://github.com/AlexPryshchenko/Landmine_detection_with_ANN/blob/main/readme_files/PMN-4.png)

They are both of 10cm diameter and have metal detonation meschanism __(1)__, explosive material __(2)__ and dielectric body __(3)__.
Main difference between them is that PMN-4 has much bigger metal meschanism than PMN-1, which definitely gives much more stronger reflection of reflected field.

Also it is important to take into account some false objects in real ground, like metal cans. Thus metal cans in a different shapes were placed to the ground:

![cans](https://github.com/AlexPryshchenko/Landmine_detection_with_ANN/blob/main/readme_files/cans.png)

Fist one is a can without the cap, second one - can with opened cap and the third - can with closed cap, which forms a slot between its body and its cap.

These 3 forms were performed in 2 different sizes, thus we have __6 unique cans__ as a false objects for recognition problem. It brings this reserch closer to real conditions. 

Together with PMN-1 and PMN-4 we have obtained __8 objects for recognition.__

These 8 objects can be located in different distances from irradiation antenna. They are from 0 to 35 cm with step of 5. Thus there are __8 possible distances__ in the grond.

### __Neural network structure__
---

In sum we have 8x8=64 possible cases that we want our neural network to recognize. Also it need to be an indication on absence of object. Thus __we need 65 output neurons__ for neural network.

As for the input data, there were made a lot of modeling to obtain these 65 cases. Every case consists of 4 signals of 769 time points for each. (time step = 0.01 ns. Thus 769 values of electric field in 7.69 ns). After stitching we have 769x4=3076 values for one case. And 769x6=4614 values after compensation of low-amplitude component with "four_to_six" function. Thus __4614 is an input shape for neural network.__

And thus train_data.txt array has size of 4614x65. 

All 65 cases are placed in a simple way of decreasing of distance, but you don't have to wory about remebering the decryption of array's columns, because in this project you can directly set a desired object and position with the help of 'set_index' function in ANN_testing.py / ANN_testing_with_animation.py.

### __Testing data__
---

Neural network can be tested on noised training signals. You can see how the training signal distorts with applying an SNR below:

![noise_gif](readme_photos/show_snr.gif)

Black curve - training signal. Red curves - testing signals with different SNRs.

SNR (Signal-to-noise ration) is a is a measure used in science and engineering that compares the power of a desired signal to the power of background noise. It can be expressed by formula:

$$
SNR=\frac{P_{signal}}{P_{noise}}
$$

Noise generation occures in "add_noise" function from data_preprocessing.py file.

You can set a one SNR value in ANN_testing.py, or a range of SNR values in ANN_testing_with_animation.py and get this animation: 

![rusult](readme_photos/result.gif)

Here you can see a signal distortion, 2D visualization which serves as interpretaion of 65 output neurons and 3D bar figure for more detailed look of appearing errors. It is very usefull for fast estimation of neural network effectivness.

Download and try it :)
