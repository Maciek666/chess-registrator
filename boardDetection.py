import cv2
import numpy as np
import imutils

filename = 'board.jpg'
img = cv2.imread(filename, 0)
x = 500
img = cv2.resize(img, (x, int(np.size(img, 0) * x / np.size(img, 1))), interpolation=cv2.INTER_AREA)

# blurowanie, binaryzacja otsu

blur = cv2.GaussianBlur(img, (5, 5), 0)

ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# th2 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
#                            cv2.THRESH_BINARY, 11, 7)

# cv2.imshow('otsu', th)
edges = cv2.Canny(blur, 0, ret)
cv2.imshow('otsu', th)
cv2.imshow('canny', edges)
mg_dilation = cv2.dilate(edges, np.ones((5, 5,), np.uint8), iterations=1)
cv2.imshow('dilation', mg_dilation)

# cnts = cv2.findContours(mg_dilation.copy(), cv2.RETR_EXTERNAL,
#                        cv2.CHAIN_APPROX_SIMPLE)
cnts = cv2.findContours(edges.copy(), cv2.RETR_TREE,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    if len(approx) == 4:
        c = c.astype("float")
        c = c.astype("int")
        cv2.drawContours(img, [c], -1, (0, 255, 0), 2)

cv2.imshow('shape', img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
