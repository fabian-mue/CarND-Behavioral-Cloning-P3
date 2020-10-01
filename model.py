# Model Code of Behavioral Cloning Project

import os
import csv
import cv2
import numpy as np
import sklearn
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D, Cropping2D, Dropout
import matplotlib.pyplot as plt

BATCH_SIZE = 32
EPOCHS = 6
KEEP_PROB = 0.5

img_size = [160, 320, 3]

'''
Generator to provide input data for training and validation;
Code provided by the lecture
'''
def generator(_samples, batch_size=32):
    num_samples = len(_samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(_samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = _samples[offset:offset + batch_size]

            images = []
            angles = []
            correction = 0.2
            steering_bias = [0, correction, - correction]
            for batch_sample in batch_samples:
                angle = float(batch_sample[3])
                # 0: image from center camera, 1: image from left camera, 2: image from right camera
                # Create additional training data by using images from left and right camera
                for i in range(0, 3):
                    name = './data/IMG/' + batch_sample[i].split('/')[-1]
                    image = mpimg.imread(name)
                    images.append(image)
                    corrected_angle = angle + steering_bias[i]
                    angles.append(corrected_angle)

                    # Create additional training data by flipping training images
                    images.append(cv2.flip(image, 1))
                    angles.append(corrected_angle * (- 1.0))

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


'''
Generates network architecture;
Chosen architecture: NVIDIA net; 
Code provided by the lecture
'''
def generate_network_architecture(_model):
    # Preprocess incoming data, centered around zero with small standard deviation
    _model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=(img_size[0], img_size[1], img_size[2])))
    _model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(img_size[0], img_size[1], img_size[2])))
    _model.add(Convolution2D(24, kernel_size=(5, 5), strides=(2, 2), activation="relu"))
    _model.add(Convolution2D(36, kernel_size=(5, 5), strides=(2, 2), activation="relu"))
    _model.add(Convolution2D(48, kernel_size=(5, 5), strides=(2, 2), activation="relu"))
    _model.add(Convolution2D(64, kernel_size=(3, 3), strides=(2, 2), activation="relu"))
    _model.add(Convolution2D(64, kernel_size=(3, 3), strides=(2, 2), activation="relu"))
    _model.add(Flatten())
    _model.add(Dense(100))
    _model.add(Dropout(KEEP_PROB))
    _model.add(Dense(50))
    _model.add(Dropout(KEEP_PROB))
    _model.add(Dense(10))
    _model.add(Dropout(KEEP_PROB))
    # final layer outputs one value due to this is a regression problem (instead of classification)
    _model.add(Dense(1))

    return _model


'''
Plots the loss of training and validation data for each epoch;
Code provided from lecture
'''
def plot_loss_information(_history_object):
    # print the keys contained in the history object
    print(_history_object.history.keys())

    # plot the training and validation loss for each epoch
    plt.plot(_history_object.history['loss'])
    plt.plot(_history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()


# Read in training data and separate in training and validation data set (80%, 20%)

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # skip the first row (headers) of the csv file
    for line in reader:
        samples.append(line)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print(len(train_samples))

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

# Create models and add architectural design
model = Sequential()
model = generate_network_architecture(model)

# Define loss function and optimizer
model.compile(loss='mse', optimizer='adam')

# Train model with defined training and validation data set
history_object = model.fit_generator(train_generator, steps_per_epoch=np.ceil(len(train_samples) / BATCH_SIZE),
                    validation_data=validation_generator,
                    validation_steps=np.ceil(len(validation_samples) / BATCH_SIZE), epochs=EPOCHS, verbose=1)

# Show loss information
plot_loss_information(history_object)

# Save the model
model.save('model.h5')
print("Model is saved")

# Show the summary of the model
model.summary()

