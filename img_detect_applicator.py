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

def process_img(image):
    simplifiedImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    simplifiedImg = cv2.Canny(simplifiedImg, threshold1=200, threshold2=300)
    vertices = np.array([[0, 280], [0, 150], [400, 150], [400, 280], [800, 280], [800, 150]])
    simplifiedImg = roi(simplifiedImg, [vertices])
    return simplifiedImg

# def enemy_detection(image):
#     subtractor = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=25, detectShadows=True)
#     img_mask =subtractor.apply(image)
#     return img_mask


subtractor = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=25, detectShadows=True)
template = cv2.imread('C:/Users/scowt/Desktop/delete me/grineer6.png', 0)
w, h = template.shape[::-1]
cv2.imshow('temp', template)

last_time = time.time()
while(True):
    warframe_screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))
    warframe_screen_gray = cv2.cvtColor(warframe_screen, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(warframe_screen_gray, template, cv2.TM_CCOEFF_NORMED)
    threshhold = 0.5
    loc = np.where(res >= threshhold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(warframe_screen, pt, (pt[0]+w, pt[1]+h), (0, 255, 255), 5)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(warframe_screen, 'corpus_grunt', (350, 450), font, 1, (200, 255, 255), 1, cv2.LINE_AA)
    # screen = subtractor.apply(warframe_screen)
    # screen = enemy_detection(warframe_screen)

    cv2.imshow('warframe_copy', warframe_screen)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
    print('loop took {} seconds'.format(time.time()-last_time))
    last_time = time.time()
