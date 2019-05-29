import numpy as np
from PIL import ImageGrab
import cv2
import time

from directkeys import ReleaseKey,PressKey, W, A , S, D

 #TODO Finish dis pls. warframe

def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

def roi2(image):
   rect = cv2.rectangle(image, (0, 280), (0, 150), (0, 255, 255), 3)
   return rect

def process_img(image):
    simplifiedImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    simplifiedImg = cv2.Canny(simplifiedImg, threshold1=200, threshold2=300)
    vertices = np.array([[0, 280], [0, 150], [400, 150], [400, 280], [800, 280], [800, 150]])
    simplifiedImg = roi(simplifiedImg, vertices)
    return simplifiedImg

# def enemy_detection(image):
#     detectedimg =
# for i in list(range(4))[::-1]:
#     print(i+1)
#     time.sleep(1)

last_time = time.time()
while(True):
    warframe_screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))
    screen = process_img(warframe_screen)
    # print('down')
    # PressKey(W)
    # time.sleep(3)
    # print('up')
    # ReleaseKey(W)
    cv2.imshow('warframe_copy', screen)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
    print('loop took {} seconds'.format(time.time()-last_time))
    last_time = time.time()





