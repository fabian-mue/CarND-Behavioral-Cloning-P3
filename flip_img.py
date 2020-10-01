import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

name = './right_2016_12_01_13_39_24_891.jpg'
image = mpimg.imread(name)
f_image = cv2.flip(image, 1)
plt.imshow(f_image)
plt.waitforbuttonpress()




