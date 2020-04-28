import math

import cv2
import numpy as np


class Detector:
    def show_image(self, image):
        scale_percent = 100

        # calculate the 50 percent of original dimensions
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)

        # dsize
        dsize = (width, height)
        normalsize = cv2.resize(image, dsize)
        cv2.imshow('imgage', normalsize)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()

    def prepare(self, image):
        blur = cv2.GaussianBlur(image, (5, 5), 0)
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpen = cv2.filter2D(blur, -1, sharpen_kernel)
        ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        canny = cv2.Canny(blur, 0, ret)
        kernel = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(canny, kernel, iterations=1)

        # cv2.imshow('image', blur)
        # cv2.waitKey()
        return dilation

    def draw_squares(self, image, cnts):
        min_area = 4500
        max_area = 5200
        squares = []
        for c in cnts:
            approx = cv2.approxPolyDP(c, 0.05 * cv2.arcLength(c, True), True)
            area = cv2.contourArea(c)
            # if min_area < area < max_area:
            if 3 < len(approx) <= 5 and (min_area < area < max_area or area > 60 * max_area):
                # x, y, w, h = cv2.boundingRect(approx)
                # cv2.rectangle(image, (x, y), (x + w, y + h), (252, 186, 3), 2)
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
                squares.append(box)
        cv2.imshow('image', image)
        cv2.waitKey()

    def find_square(self, image):
        blur = cv2.medianBlur(image, 5)
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpen = cv2.filter2D(blur, -1, sharpen_kernel)

        thresh = cv2.threshold(sharpen, 160, 255, cv2.THRESH_BINARY_INV)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        cnts = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        self.draw_squares(image, cnts)

    def find_square_2(self, image):
        prepared = self.prepare(image)
        cnts, hier = cv2.findContours(prepared, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        print(hier)
        #cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        min_area = 4500
        max_area = 5200
        squares = []
        for c in cnts:
            approx = cv2.approxPolyDP(c, 0.1 * cv2.arcLength(c, True), True)
            area = cv2.contourArea(c)
            # print(area)
            # if min_area < area < max_area:
            if 3 < len(approx) <= 4 and (min_area < area < max_area or area > 60 * max_area):
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                squares.append(box)

        self.draw_squares(image, cnts)
        return squares


if __name__ == '__main__':
    d = Detector()
    image = cv2.imread('D:\Programming\python\chess-registrator\photos\ze_stojaka_1.jpg', 0)
    # img = d.prepare(image)
    sq = d.find_square_2(image)
    #print(sq)
    # d.show_image(img)
