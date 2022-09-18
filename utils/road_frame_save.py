import cv2
import sys
import time

def save_image_and_steering_angle():
    frame_rate = 3
    prev = 0
    guardar = False
    cap = cv2.VideoCapture(0)

    try:
        i = 0
        while(True):
            time_elapsed = time.time() - prev
            _, frame = cap.read()

            cv2.imshow('Frame', frame)

            if guardar:
                if time_elapsed > 1./frame_rate:
                    prev = time.time()
                
                    cv2.imwrite("frame_{}.png".format(i), frame)
                    i += 1

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if key == 32:
                print("Guardar:",guardar)
                guardar = not guardar

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    save_image_and_steering_angle()
