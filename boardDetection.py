import cv2
import numpy as np

filename = 'board.jpg'
img = cv2.imread(filename, 0)
x = 800
img = cv2.resize(img, (x, int(np.size(img, 0) * x / np.size(img, 1))), interpolation=cv2.INTER_AREA)

# blurowanie, binaryzacja otsu

blur = cv2.GaussianBlur(img, (5, 5), 0)

ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# th2 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
#                            cv2.THRESH_BINARY, 11, 7)

cv2.imshow('otsu', th)

if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
