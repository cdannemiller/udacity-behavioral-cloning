# SDC: Behavioral Cloning Project

### Software Used

* Anacoda: 4.3.21
* Python:  3.5.4
* TensorFlow: 1.4.0
* Keras: 2.0.9

### Data gathering

At first I simply went aound the track once, however, this wasn't enough data. In order to increase the amount of data captured, I used a version of the simulator that records 50 frames a second. The data stil had an issue plotting the data in a histogram made it apparent what the issue was:

Most of the data has the car not turning either to the left or the right, this makes sense as most driving is done in a strait line. However, this does bias the network tword driving in a strait line. There is code in model.py to remove bins that have a large bais, this makes the distribution of the training data more guassian.

In order to augment the data captured three transformations were done. The first is to simply invert the image and the steering angle, as suggested in the course videos. The second it to use the left and right images like in the course videos, however, I found a offset of 0.15 worked better for my system than the value of 0.2. Finally since the images are converted into YUV space, adjusting the brightness of the images is trivial, and therefore is done during training to prevent the learning for getting a bais from a given light level.

I also cropped the image in a similiar fashion to the suggestion of the course video. I also found that the video could be shrunk by 50% without loss of accuracy, this not only makes training faster but also prediction.

### Network Design

For the design of the network I started with the design described by NVIDIA's self driving team. https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/. This design was modified slightly, I used five convolutional layers with a max pooling after each layer. Then I used three fully connectected layers that were 100, 50, and 10 nodes respectivly. After each of the layers a leakly ReLU was used to allow the system to learn non-linear functions.

In order to help address overfitting 50% dropouts are placed between the fully connected nodes.

### Color Issues
One issue I encounted is despite the fact that the training seemd to go well it did not transfer to the client. I noticed that the red and blue channels were swapped between the traning that was done in model.py and using the training model to drive the car in drive.py. To allivate this issue, swapped the red and blue channels in drive.py. After making this change the car was finally able to drive sucessfully around the track.

### Training and Validation
The left, right and center images are used for training. However, the stearing for the left and right images is ajdusted by a fudge factor, and cannot be considered representative of the true data. Therefore the left and right images are not included in the validation set. The training and valdiation data are split between 80% and 20% of the original data respecitivly. There is not testing set in the case as the actual simulation is used as the testing set.

The system was training for give Epochs at this point the validation loss did not seem to be going down a


### Reflections

I treid several different things to get the second course working. More data agumentation, learning both the stearing and the speed inputs, training with even more data. In the end I ran out of time and have several thoughts about how to improve the system to get the second course predicted

* Usage of a LSTM network
* More training data
* Better filtering of excesss data.
* Adjust both the brightness and contrast of the image, as the second track as more lighting varations than the first.

The physics of the game world is not very reflective of real world physics, as it the car as a rigid object. It doesn't properly simulate the suspension of a car has. This causes the car to fly off the road in situations that do not occur in real life. A real car would also hug the turns more easily than the simulation. A possilbe solution to this problem is to use a game that can be scriptted with like GTA V. 

As a gamer initial I had the natural desire to train the car to find optimal racing lines, however, not only is this not a tyipical driver behaves, it infact caused the car to leave the road more. It would, however, be interested to see if it is possilbe to train a model in the setting of competive racing.
