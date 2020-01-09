import cv2
import numpy as np
import imutils

filename = 'board.jpg'
img = cv2.imread(filename, 0)
x = 300
img = cv2.resize(img, (x, int(np.size(img, 0) * x / np.size(img, 1))), interpolation=cv2.INTER_AREA)

# blurowanie, binaryzacja otsu

blur = cv2.GaussianBlur(img, (5, 5), 0)

ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow('otsu', th)

# th2 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
#                            cv2.THRESH_BINARY, 11, 7)

edges = cv2.Canny(blur, 0, ret)
cv2.imshow('canny', edges)
mg_dilation = cv2.dilate(edges, np.ones((5, 5,), np.uint8), iterations=1)
cv2.imshow('mg_dilation', mg_dilation)

# cnts = cv2.findContours(mg_dilation.copy(), cv2.RETR_EXTERNAL,
#                        cv2.CHAIN_APPROX_SIMPLE)
cnts = cv2.findContours(mg_dilation.copy(), cv2.RETR_TREE,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
imgEdges = cv2.cvtColor(edges, cv2.COLOR_RGB2BGR)

for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    if len(approx) == 4:
        c = c.astype("float")
        c = c.astype("int")
        cv2.drawContours(imgEdges, [c], -1, (0, 255, 0), 2)

cv2.imshow('edges', imgEdges)

# Probabilistic Line Transform
linesP = cv2.HoughLinesP(mg_dilation, 1, np.pi / 180, 150, None, 20, 10)
# Draw the lines
cdst = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv2.line(cdst, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 1, cv2.LINE_AA)

cv2.imshow('HLP', cdst)

cdst = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
lines = cv2.HoughLines(mg_dilation, 1, np.pi / 180, 50, None, 0, 0)
for rho, theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    cv2.line(cdst, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imshow('houghlines', cdst)

# Detector parameters
blockSize = 2
apertureSize = 3
k = 0.04
# Detecting corners
dst = cv2.cornerHarris(th, blockSize, apertureSize, k)
# Normalizing
dst_norm = np.empty(dst.shape, dtype=np.float32)
cv2.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
dst_norm_scaled = cv2.convertScaleAbs(dst_norm)
# Drawing a circle around corners
for i in range(dst_norm.shape[0]):
    for j in range(dst_norm.shape[1]):
        if 80 > int(dst_norm[i, j]) > 50:
            cv2.circle(dst_norm_scaled, (j, i), 5, (0), 2)
# Showing the result
cv2.imshow('corners_window', dst_norm_scaled)

if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()

