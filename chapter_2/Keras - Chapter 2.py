#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Get the MNIST dataset, in the form of four Numpy arrays

from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# In[2]:


# Make sure that the data imported is correct

print(train_images.shape)
print(len(train_labels))
print(train_labels)

print(test_images.shape)
print(len(test_labels))
print(test_labels)


# In[3]:


# Display an image using Matplotlib

print(train_images.ndim)
print(train_images.shape)
print(train_images.dtype)

digit = train_images[4]

import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()


# In[4]:


# Import the modules that we will need

from keras import models
from keras import layers


# In[5]:


"""
The core building block of neural networks is the layer, a data-processing 
module that you can think of as a filter for data. Layers extract representations
out of the data fed into them - hopefully ones that are meaningful for the 
problem at hand. Most of deep learning consists of chaining together simple 
layers that will implement a form of progressive data distillation.
"""

"""
Here, our network consists of a sequence of two Dense layers, which are densely
connected (aka fully connected) neural layers. The second (and last) layer is a
10-way softmax layer, which means it will return an array of 10 probability 
scores (summing to 1). Each score will be the probability that the current digit
image belongs to one of our 10 digit classes.
"""

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))


# In[6]:


"""
To make the network ready for training, we need to pick three more things, as
part of the compilation step:

A loss function: How the network will be able to measure its performance on 
the training data, and thus how it will be able to steer itself in the right
direction.

An optimizer: The mechanism through which the network will update itself based
on the data it sees and its loss function.

Metrics to monitor during training and testing: Here, we'll only care about
accuracy (the fraction of the images that were correctly classified).
"""

# The compilation step

network.compile(optimizer='rmsprop', loss='categorical_crossentropy',                 metrics=['accuracy'])


# Preparing the image data

"""
Before training, we'll preprocess the data by reshaping it into the shape the
network expects and scaling it so that all values are in the [0, 1] interval.
Previously, our training images, for instance, were stored in an array of shape
(60000, 28, 28) of type unit8 with values in the [0, 255] interval. We transform
it into a float32 array of shape (60000, 28 * 28) with values between 0 and 1.
"""

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255


# Preparing the labels

"""
We also need to categorically encode the labels.
"""

from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# In[7]:


"""
We're now ready to train the network, which in Keras, is done via a call to 
the network's fit method - we fit the model to its training data.
"""

network.fit(train_images, train_labels, epochs=5, batch_size=128)


# In[8]:


"""
We quickly reach an accuracy of 0.989 on the training data. Now let's check
that the model performs well on the test set, too.
"""

"""
The test-set accuracy ends up being lower than the training set accuracy. 
This gap between training accuracy and test accuracy is an example of 
overfitting: the fact that machine-learning models tend to perform worse on
new data than on their training data.
"""

# Check that the model performs well on the test set, too

test_loss, test_acc = network.evaluate(test_images, test_labels)

print('test_acc:', test_acc)


# In[ ]:




