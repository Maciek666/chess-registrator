from unittest import TestCase

import cv2
import numpy as np

from scripts.detector import Detector


class TestDetector(TestCase):
    def test_fields(self):
        path = 'D:\Programming\python\chess-registrator\photos\ze_stojaka_4.jpg'
        image_ = cv2.imread(path, 0)
        detector = Detector()
        cropped = detector.get_board(image_)
        fields = detector.get_fields(cropped)
        self.assertEqual(len(fields), 8, ' nie znaleziono 8 rzedow')
        for rzad in fields:
            self.assertEqual(len(rzad), 8, 'nie poprawna ilosc pol w rzedzie')

    def test_lines(self):
        path = 'D:\Programming\python\chess-registrator\photos\ze_stojaka_4.jpg'
        image_ = cv2.imread(path, 0)
        detector = Detector()
        board_image = detector.get_board(image_)
        clache = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl_board = clache.apply(board_image)
        image = detector._prepare(cl_board)
        h8, width = board_image.shape[0], board_image.shape[1]
        cdst = np.zeros((h8, width, 3), np.uint8) + 250
        # cdst = cv2.cvtColor(board_image, cv2.COLOR_GRAY2BGR)
        detected_lines = detector._prepare_lines(image)
        poziome, pionowe = detector._group_line(detected_lines)

        pionowe, dist_pion = detector._filter_vertical_lines(pionowe, width)
        poziome, dist_poz = detector._filter_horizontal_lines(poziome, h8)

        pionowe.sort(key=lambda x: x[0][0])
        poziome.sort(key=lambda x: x[0][1])
        # self._lines_reconstruction(pionowe, dist_pion, 0)
        # self._lines_reconstruction(poziome, dist_poz, 1)
        detector._horizontal_lines_reconstruction(poziome, dist_poz, h8)
        detector._vertical_lines_reconstruction(pionowe, dist_pion, width)

        self.assertEqual(len(pionowe), 9, 'pionowe linie != 9')
        self.assertEqual(len(poziome), 9, 'poziome linie != 9')
