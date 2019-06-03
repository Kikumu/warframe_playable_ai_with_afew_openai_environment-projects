import numpy as np   #array operations
import matplotlib.pyplot as plt  #show image
import os  #iterate through directory
import cv2 #image operations
import pickle
import time

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard

DATADIR = "C:/Users/scowt/Desktop/grunts_warframe"
CATEGORIES = ["Corpus", "Grineer"]

 #grab original images
for category in CATEGORIES:
    path = os.path.join(DATADIR, category) #path to cats/dogos dir
    for img in os.listdir(path): #itr thru imgs
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)#convert img to grayscale. first converted to array then to grayscale. rgb is 3 times bigger than grayscale
        break
        break

 #reshape image
IMG_SIZE = 80
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) #image operations as seen here xd
plt.imshow(new_array, cmap='gray')
plt.show()

  #train datasets

training = []  # training data set
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)  # path to cats/dogos dir
        class_num = CATEGORIES.index(
            category)  # we cant map shit to string we gonna match em to a numerical value eg 0 or 1. has to match to a number
        for img in os.listdir(path):  # itr thru imgs
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert img to grayscale. first converted to array then to grayscale. rgb is 3 times bigger than grayscale
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training.append([new_array, class_num])  # append data/add to end
            except Exception as e:
                pass


create_training_data()
# nb indentation matters

print(len(training))

import random #shuffle data for better learning
random.shuffle(training)

for sample in training[:10]: #do until 10?
    print(sample[1])#check if labels correct? (actual img array)

X = [] #feature?
y = [] #label?

for features, label in training:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)



pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

# pickle_in = open("X.pickle","rb")
# X = pickle.load(pickle_in)

 # training model

NAME = "warframe-grunts-trial-64x2-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

x = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

X = x / 255.0  # 255 caause of color scheme rangers from 0 - 255. is this our filter?

model = Sequential()
model.add(
    Conv2D(69, (3, 3), input_shape=X.shape[1:]))  # is 69 units our filter? its on a 3 by 3 matrix! this is our filter
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))  # get largest value

model.add(Conv2D(69, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # dense layer accepts only 1D ITS CONVERTED HERE
model.add(Dense(69))

# output layer
model.add(Dense(1))
model.add(Activation('sigmoid'))

#binary cause 2 things (cats and dogs)
model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=10, validation_split=0.1,callbacks=[tensorboard]) # how many at a time do you wanna pass through each data depends on data size
model.save('64x83-WFG.model')
# tensorboard --logdir=foo:C:/Users/scowt/PycharmProjects/untitled1/logs
graph_def = graph_pb2.GraphDef()

