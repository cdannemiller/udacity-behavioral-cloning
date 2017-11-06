# Udacity SDC: Behavioral Cloning 
# Christopher Dannemiller
import cv2
import csv
import numpy as np
import os
import progressbar
import sklearn
from enum import Enum
from random import shuffle
import utils
import matplotlib.pyplot as plt

# Hyper parameters.
steering_adjust = 0.15                      # The amount of adjustment to place on the controller for the left & right images.
batch_size = 256                            # Number of images to train in each batch.
epochs = 5                                  # Number of epochs to train for.

samples = []

def add_capture_set(path):
    '''
    Imports images and steering data from a single directly on the local computer.
    '''
    print("Importing image collection from: {}".format(path))

    with open(path + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)

        for line in reader:
            # Convert paths to be relative to the running directory.
            line[0] = path + os.sep + 'IMG' + os.sep + line[0].split(os.sep)[-1]
            line[1] = path + os.sep + 'IMG' + os.sep + line[1].split(os.sep)[-1]
            line[2] = path + os.sep + 'IMG' + os.sep + line[2].split(os.sep)[-1]
            line[3] = float(line[3])
            samples.append(line)

def generate_data(sample, validation):
    '''
    Generates data for a single image. 
    
    If the it the data is not being validated, the brightness is changed randomly. In addition,
    The image and steering angle is flipped to create two images from one.
    '''
    image = utils.process_image(cv2.imread(sample[0]))

    if not validation:
        image = utils.adjust_brigtness(image, np.random.randint(-25, 25))

    # Return both the image and its inverse
    imgs = [image, cv2.flip(image, 1)]
    out = [sample[1:2],[-sample[1]]]
    return imgs, out        

# These paths are relative to the running directory.
add_capture_set('cap/6')
add_capture_set('cap/7')
add_capture_set('cap/8')

# Split the data into a training and validation set.
from sklearn.model_selection import train_test_split
train_samples_raw, validation_samples_raw = train_test_split(samples, test_size=0.2)

train_samples = []
validation_samples = []

# For the training data create a image for the left, right and center images. However the creation of this data is for
# training purposes only and is not the ground truth. Therefore the usage of the left and right images is restricted to the
# training set.
for sample in train_samples_raw:
    train_samples.append([sample[0], sample[3]])
    train_samples.append([sample[1], sample[3] + steering_adjust])
    train_samples.append([sample[2], sample[3] - steering_adjust])

for sample in validation_samples_raw:
    validation_samples.append([sample[0], sample[3]])

gens_per_sample = len(generate_data(train_samples[0], False)[0])
print("Number of training samples per image: ", gens_per_sample)

def generator(samples, batch_size=32, validation = False):
    '''
    '''
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)

        sampleIdx = 0
        images = []
        steerings = []

        while sampleIdx < len(samples):
            while sampleIdx < len(samples) and len(images) < batch_size:
                sample = samples[sampleIdx]
                imgs, adjusts = generate_data(sample, validation)
                images = images + imgs
                steerings = steerings + adjusts
                sampleIdx = sampleIdx + 1
            X_train = np.array(images[:batch_size])
            y_train = np.array(steerings[:batch_size])
            images = images[batch_size:]
            steerings = steerings[batch_size:]
            if images is None: images = []
            if steerings is None: steerings = []
            yield sklearn.utils.shuffle(X_train, y_train)

        if len(images) != 0:
            X_train = np.array(images[:batch_size])
            y_train = np.array(steerings[:batch_size])
            yield sklearn.utils.shuffle(X_train, y_train)


print("Bringing up TensorFlow....")

# The following code dodges a CUDA_ERROR_OUT_OF_MEMORY error on my machine.
import tensorflow as tf
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=config)
sess.as_default()

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Dropout, MaxPooling2D, Cropping2D
from keras.layers import ELU, LeakyReLU

model = Sequential()
model.add(Lambda(lambda x : (x / 255.0) - 0.5, input_shape=(32,160,3)))
model.add(Conv2D(24,(5,5),activation="relu",strides=(1,1), padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(36,(5,5),activation="relu",strides=(1,1), padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(48,(5,5),activation="relu",strides=(1,1), padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(64,(3,3),activation="relu",strides=(1,1), padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(64,(3,3),activation="relu",strides=(1,1), padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Flatten())
model.add(Dense(100))
model.add(LeakyReLU())
model.add(Dense(50))
model.add(LeakyReLU())
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(LeakyReLU())
model.add(Dropout(0.5))
model.add(Dense(1))

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size, validation = False)
validation_generator = generator(validation_samples, batch_size=batch_size, validation = True)
        
model.compile(loss = 'mse', optimizer = 'adam')
model.fit_generator(train_generator,\
          steps_per_epoch=float(len(train_samples) * gens_per_sample)/float(batch_size), \
          validation_data=validation_generator, \
          validation_steps=float(len(validation_samples) * gens_per_sample)/float(batch_size), \
          epochs=epochs\
         )

print("Saving Model...")
model.save('model.h5')
print("Saved")
#    
