import chess
import cv2
from scripts.detector import Detector
from sklearn.cluster import KMeans


class Tracker:
    def __init__(self, fields: list):
        """
        :param fields: list of triplets, first corner coordinates, second corner coordinates, field name
            cordiantes(y,x)
        """
        self.board = chess.Board()
        self.fields = fields

    def compare(self, board_image_1, board_image_2):
        board_image_2 = cv2.blur(board_image_2, (5, 5))
        board_image_1 = cv2.blur(board_image_1, (5, 5))
        cv2.imshow('before', board_image_1)
        cv2.imshow('after', board_image_2)

        frameDelta = cv2.absdiff(board_image_1, board_image_2)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        cv2.imshow('',frameDelta)
        cv2.imshow('',thresh)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
        points = []
        for c in cnts:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                points.append([cX, cY])

        kmeans = KMeans(n_clusters=2, random_state=0).fit(points)
        centers = kmeans.cluster_centers_
        return centers

    def find_move(self, centers):
        """

        :param  centers: [x, y]:
                fields: [y, x]
        :return:
        """
        move = []
        for line in self.fields:
            for field in line:
                a = field[1]
                b = field[0]
                for c in centers:
                    if a[1] <= c[0] <= b[1] and a[0] <= c[1] <= b[0]:
                        print(field[2])
                        move.append(field[2].lower())
        if len(move) != 2:
            raise Exception('crush')
        return move

    def cropp_and_rotate_like_first_scene(self, detector: Detector, image):
        height, width = image.shape[0], image.shape[1]
        rot_matrix = cv2.getRotationMatrix2D(detector.center, detector.angle, 1)
        rotated = cv2.warpAffine(image, rot_matrix, (width, height))
        cropped = cv2.getRectSubPix(rotated, detector.size, detector.center)
        return cropped


if __name__ == '__main__':
    path = 'D:\Programming\python\chess-registrator\\after.jpg'
    path_2 = 'D:\Programming\python\chess-registrator\\before.jpg'

    image_ = cv2.imread(path, 0)
    image_2 = cv2.imread(path_2, 0)
    detector = Detector()
    cropped = detector.get_board(image_)
    fields = detector.get_fields(cropped)
    tracker = Tracker(fields)
    cropped_2 = tracker.cropp_and_rotate_like_first_scene(detector, image_2)
    move = tracker.compare(cropped, cropped_2)
    tracker.find_move(move)
    cv2.waitKey()
