# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

In this project I orientated my CNN model by the NVIDIA network for self driving cars (https://arxiv.org/pdf/1604.07316v1.pdf).

The final model starts with a normalization layer which pre-processes the incoming image data with the 160 rows, 320 columns and 3 channels to center them around zero with small standard deviation (code line 24). Next a cropping layer is used to crop not relevant parts of the image to focus on the useful parts of the image for training and prediction (code line 25). 50 rows pixels from the top of the image and 20 rows pixels from the bottom of the image have been removed.
Afterwards the model includes 5 convolution layer with 2x2 strides and relu activation functions to introduce nonlinearity. The first three convolution layers use a 5x5 kernel and the other two use 3x3 kernels. The convolution layers are implemented in the lines 66 - 70. After the last convolution the output feature map contains 64 filters and is flattened to input it into the upcoming four fully-connected-layers (lines 72, 74, 76, 79).  The last fully-connected-layer output only one value (representing the predicted steering angle) because this is a regression problem. Additionally, except for the last one after each fully-connected-layer a dropout layer with a keep probability of 0.5 is introduced in order to regularize the data and prevent overfitting (lines 73, 75, 77).

#### 2. Attempts to reduce overfitting in the model

As already mentioned the model contains dropout layers in order to reduce overfitting (lines 73, 75, 77). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 113-114). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer with the default learning rate of 0.001, so the learning rate was not tuned manually (model.py line 121).

#### 4. Appropriate training data

For training, the provided sampled data has been chosen. 
Aside to the center image I also used the right and left camera images for training (lines 41-46). This was especially helpful to improve the driving maneuver of the car in situation where it is on the very left/right side of the road.
Additionally, I augmented all of these images by flipping them horizontally to retrieve a larger dataset with various images. This technique also supported handling the left turn bias issue because now there are also images where the car is "driving" clockwise.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to find a model which has enough convolutions and the right pre-processing layers to learn efficiently extracting the correct features for prediction from the training data but also to prevent overfitting at the same time.

My first step was to use a convolution neural network model similar to the LeNet. I thought this model might be appropriate because it already worked very fine in the Traffic Sign Classification project. However, I also tried the architecture of the NVIDIA model and noticed that I could achieve better results with this model.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. By checking the mean squared error on the validation set, I figured out that 6 is a sufficient number of epochs to have good prediction results without overfitting the network.
Still, to combat the overfitting,  additionally I modified model and introduced dropout layers.
Additionally, the right and left camera images have been added to the training data set and augmented to make the model able to react on more different situations and especially when it is driving on the left or right side of the road.

The final step was to run the simulator to see how well the car was driving around track one. Using the described and trained model, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 62-81) consisted of a convolution neural network orientated at the NVIDIA model (https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). A detailed description of the final model can be found in the section "An appropriate model architecture has been employed" in the chapter "Model Architecture and Training Strategy" .

Here is a visualization of the mentioned NVIDIA model:

![](/home/fabian/CarND-Behavioral-Cloning-P3/architecture.png)

#### 3. Creation of the Training Set & Training Process

To train the network, I used the provided sampled training dataset.
After reading in the training data from the .csv file I separated it into training and validation data set.  At this I put 20% of the data into a validation set. 

In order to have versatile training data, I used camera images from the left, center and right cameras. Left and right cameras images have been used to improve the behavior of the vehicle when it is almost too far on the right or left side of the road, that it reacts and steers in the center of the road. For left images the labeled steering angle was increased by 0.2 and for right camera images the angle was decreased by 0.2.

![alt text][image3]
![alt text][image4]
![alt text][image5]

To avoid a high memory consumption, a generator is used to provide training data with a batch size of 32. Hence, only these images had to be stored. During the generation of the training data batches, the training data was shuffled.
Furthermore, in the generator the images are augmented. To augment the data set, I also flipped images and angles thinking that this technique also supports handling the left turn bias issue because now there are also images where the car is "driving" clockwise. Additionally, simply more different training data could be generated. 
For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

With 3 images per sample and additional augmenting, the final training data hat a size of 38568 images.

Per iteration I preprocessed a batch of this data by normalizing the image and cropping irrelevant image parts for the prediction, so that the model can focus more on the relevant features in the image.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 6 as evidenced by a low mean squared error loss. For training, I used an adam optimizer which adapts the learning rate, so that manually training the learning rate wasn't necessary.
