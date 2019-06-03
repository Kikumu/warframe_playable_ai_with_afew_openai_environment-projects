import cv2
import tensorflow as tf
import time
import numpy as np
from PIL import ImageGrab

CATEGORIES = ["Corpus", "Grineer"]
model = tf.keras.models.load_model("64x83-WFG.model")
#using actual model on actual data basically reverse of what we did the first time, reconverting those numbers to strings
# def prepare(filepath):
#     IMG_SIZE = 80
#     img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
#     new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
#     return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
# model = tf.keras.models.load_model("64x83-WFG.model")
#in the predict one below its ALWAYS A LIST

# prediction = model.predict([prepare('C:/Users/scowt/Desktop/delete me/grineer6.png')])
# print(CATEGORIES[int(prediction[0][0])])

#from value obtained by neural net draw highlighter around the object the neural net thinks its the answer
last_time = time.time()
while(True):
    warframe_screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))  #set screen size
    warframe_screen_gray = cv2.cvtColor(warframe_screen, cv2.COLOR_BGR2GRAY)
    print(tf.VERSION)
    # res = cv2.matchTemplate(warframe_screen_gray, template, cv2.TM_CCOEFF_NORMED)
    # threshhold = 0.5
    # loc = np.where(res >= threshhold)
    # for pt in zip(*loc[::-1]):
    #     cv2.rectangle(warframe_screen, pt, (pt[0]+w, pt[1]+h), (0, 255, 255), 5)
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     cv2.putText(warframe_screen, 'corpus_grunt', (350, 450), font, 1, (200, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow('warframe_copy', warframe_screen)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
    print('loop took {} seconds'.format(time.time()-last_time))
    last_time = time.time()
