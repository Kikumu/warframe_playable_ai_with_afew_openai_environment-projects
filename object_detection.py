import numpy as np
import cv2

img = cv2.imread('C:\\Users\\scowt\\Desktop\\grunts_warframe\\corpus_grunt1.png', cv2.IMREAD_COLOR)
cv2.rectangle(img, (15, 25), (165, 370), (0, 255, 0), 4)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'enemy', (50, 390), font, 1, (200, 255, 255), 2, cv2.LINE_AA)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()