# SDC: Behavioral Cloning Project

### Software Used

* Anacoda: 4.3.21
* Python:  3.5.4
* TensorFlow: 1.4.0
* Keras: 2.0.9

### Introduction
This project is designed to have a human train a Neural Network how to drive using behavioral cloning. This is done using a provided simulator that replicates the basic elements of driving and outputs images along with input values, notably steering angle, throttle, and brake. The network is then trained to learn to predict the value of the steering angle from the image. By predicting this angle, the computer can learn to drive like its training.

To train the model `python model.py` can be invoked. However, the training data is not provided in this repo. To use the training data `python drive model.h5` can be invoked. I made minor alterations to drive.py from the class provided version. The most notable change is that I increased the speed of the car.

### Data gathering

At first I just went around the track once; however, this wasn't enough data. To increase the amount of data captured per lap, I used a version of the simulator that records 50 frames a second. In the end, the training data consisted of two laps going around the track, another two laps going around the track in the opposite direction, and finally, in the last data set I would drive the car the edge, start recording and recover the care back to the center. This last data set was critical, as my datasets for driving around the track normally were too pristine and the network did not compensate.

Most of the data have the car not turning either to the left or the right; this makes sense as most driving is done in a straight line. However, this does bias the network toward driving in a straight line. There is code in model.py to remove bins that have a large bais; this makes the distribution of the training data more Gaussian.

To augment the data captured three transformations were done. The first is to simply invert the image and the steering angle, as suggested in the course videos. The second is to use the left and right images like in the course videos, however, I found an offset of 0.15 worked better for my system than the value of 0.2. Finally, since the images are converted into YUV space, adjusting the brightness of the images is trivial, and therefore is done during training to prevent the learning for getting a bais from a given light level.

I also cropped the image in a similar fashion to the suggestion of the course video. I also found that the video could be shrunk by 50% without loss of accuracy, this not only makes training faster but also prediction.

### Network Design

For the design of the network, I started with the design described by NVIDIA's self-driving team. https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/. This design was modified slightly; I used five convolutional layers with a max pooling after each layer. Then I used three fully connected layers that were 100, 50, and ten nodes respectively. After each of the layers, a leaky ReLU was used to allow the system to learn non-linear functions.

To help address overfitting 50% dropouts are placed between the fully connected nodes.

### Color Issues
One issue I encountered is despite the fact that the training seemed to go well it did not transfer to the client. I noticed that the red and blue channels were swapped between the training that was done in model.py and using the training model to drive the car in drive.py. To alleviate, this issue swapped the red and blue channels in drive.py. After making this change, the car was finally able to drive successfully around the track.

### Training and Validation
The left, right and center images are used for training. However, the steering for the left and right images is adjusted by a fudge factor, and cannot be considered representative of the true data. Therefore the left and right images are not included in the validation set. The training and validation data are split between 80% and 20% of the original data respectively. There is not testing set in the case as the actual simulation is used as the testing set.

The system was training for five Epochs at this point the validation loss did not seem to be going down anymore.

### Reflections

I tried several different things to get the second course working. More data argumentation, learning both the steering and the speed inputs, training with even more data. In the end, I ran out of time and had several thoughts about how to improve the system to get the second course predicted

* Usage of an LSTM network
* More training data
* Better filtering of excess data.
* Adjust both the brightness and contrast of the image, as the second track as more lighting variations than the first.
* The data is heavily biased tword driving forward, the excess data could be removed.

The physics of the game world is not very reflective of real-world physics, as it the car as a rigid object. It doesn't properly simulate the suspension of a car has. This causes the car to fly off the road in situations that do not occur in real life. A real car would also hug the turns more efficiently than the simulation. A possible solution to this problem is to use a game that can be scripted with like GTA V. 

As a gamer initial I had the natural desire to train the car to find optimal racing lines, however, not only is this, not a typical driver behaves, it, in fact, caused the car to leave the road more. It would, however, be interesting to see if it is possible to train a model in the setting of competitive racing.
