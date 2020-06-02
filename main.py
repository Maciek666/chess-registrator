import chess
import cv2

from scripts.detector import Detector
from scripts.tracker import Tracker

cam = cv2.VideoCapture(0)
cam.set(3, 1024)
cam.set(4, 768)
cv2.namedWindow("test")
img_counter = 0
detector = Detector()
tracker = None
fields = None
prev = None
while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        # img_name = "opencv_frame.jpg"
        # cv2.imwrite(img_name, frame)
        # print(f'written {img_name}')
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        try:
            if fields is None:

                prev = detector.get_board(frame)
                fields = detector.get_fields(prev)
                tracker = Tracker(fields)
                detector._show_fields(fields, prev)
            else:
                current = tracker.cropp_and_rotate_like_first_scene(detector, frame)
                move_centers = tracker.compare(prev, current)
                move_name = tracker.find_move(move_centers)
                if tracker.board.is_legal(chess.Move.from_uci(str(move_name[0]).lower() + str(move_name[1]).lower())):
                    tracker.board.push_uci(move_name[0] + move_name[1])
                elif tracker.board.is_legal(chess.Move.from_uci(str(move_name[1]).lower() + str(move_name[0]).lower())):
                    tracker.board.push_uci(move_name[1] + move_name[0])
                else:
                    raise Exception('nielgalny ruch')
                prev = current

            print(tracker.board)
            cv2.imshow('cropped', prev)
            cv2.waitKey()
            #  fields = detector.  get_fields(cropped)
        except Exception as e:
            print(e)
            print('sprobuj jeszcze raz')

cam.release()
cv2.destroyAllWindows()
