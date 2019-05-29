import cv2
import tensorflow as tf
import numpy as np
from PIL import ImageGrab

CATEGORIES = ["Corpus", "Grineer"]
#using actual model on actual data basically reverse of what we did the first time, reconverting those numbers to strings
def prepare(filepath):
    IMG_SIZE = 80
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.models.load_model("64x83-WFG.model")
#in the predict one below its ALWAYS A LIST

prediction = model.predict([prepare('C:/Users/scowt/Desktop/delete me/grineer6.png')])
print(CATEGORIES[int(prediction[0][0])])

#from value obtained by neural net draw highlighter around the object the neural net thinks its the answer
threshhold = 0.3
