import cv2
import sys
import time

def save_image_and_steering_angle():
    guardar = False
    cap = cv2.VideoCapture(0)

    try:
        i = 0
        while(True):
            _, frame = cap.read()
            cv2.imshow('Frame', frame)

            if guardar:
               cv2.imwrite("frame_{}.jpeg".format(i), frame)
               i += 1
               guardar = False

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if key == 32:
                guardar = not guardar

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    save_image_and_steering_angle()
