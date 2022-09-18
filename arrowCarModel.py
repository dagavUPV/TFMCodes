import cv2
import numpy as np
import math
from tensorflow.keras.models import load_model
import random
import time
import sys

from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.adapters.common import input_size
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

_SHOW_IMAGE = False

def img_preprocess(image):
    height, _, _ = image.shape
    image = image[int(height/2):,:,:]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image = cv2.GaussianBlur(image, (3,3), 0)
    image = cv2.resize(image, (200,66))
    image = image / 255
    return image

def display_heading_line(frame, steering_angle, line_color=(0, 0, 255), line_width=5, ):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape

    steering_angle_radian = steering_angle / 180.0 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)

    cv2.arrowedLine(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)

    return heading_image


def show_image(title, frame, show=_SHOW_IMAGE):
    if show:
        cv2.imshow(title, frame)

def detect_objects(frame,interpreter,inference_size,labels, velocidad):

    min_confidence = 0.40
    num_of_objects = 3
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontColor = (255, 255, 255)

    cv2_im = frame
    cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
    cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)

    run_inference(interpreter, cv2_im_rgb.tobytes())

    objects = detect.get_objects(interpreter, min_confidence)[:num_of_objects]

    if objects:
        frame, velocidad = append_objs_to_img(cv2_im, inference_size, objects, labels, velocidad)

    frame = cv2.putText(frame, velocidad, (10,40), font, 1, fontColor, 2)

    return frame, velocidad

def append_objs_to_img(cv2_im, inference_size, objs, labels, velocidad):

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 40)
    middleText = (300, 40)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2

    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    
    for obj in objs:
        bbox = obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)

        percent = int(100 * obj.score)
        object_detected = labels.get(obj.id, obj.id)

        label = '{}% {}'.format(percent, object_detected)

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        if object_detected == "stop":
            cv2_im = cv2.putText(cv2_im, "STOP", middleText, font, fontScale, fontColor, lineType)
        elif object_detected == "crosswalk":
            cv2_im = cv2.putText(cv2_im, "PRECAUCION", middleText, font, fontScale, fontColor, lineType)
        elif object_detected == "speedlimit50":
            velocidad = "Velocidad 50"
        elif object_detected == "speedlimit60":
            velocidad = "Velocidad 60"
        elif object_detected == "speedlimit70":
            velocidad = "Velocidad 70"
        elif object_detected == "speedlimit80":
            velocidad = "Velocidad 80"
        elif object_detected == "speedlimit90":
            velocidad = "Velocidad 90"

    return cv2_im, velocidad

#video path
path = sys.argv[1]

#road model
model_path='/home/pi/piCar/DeepPiCar/driver/code/models/road_navigation_final.h5'
curr_steering_angle = 90
model_road = load_model(model_path, compile = False)

#signs model
model_sign='/home/pi/piCar/DeepPiCar/driver/code/models/efficientdet-lite-car_signs-eDet1.tflite'
label='/home/pi/piCar/DeepPiCar/driver/code/models/car_signs-labels-eDet1.txt'

labels = read_label_file(label)
interpreter = make_interpreter(model_sign)
interpreter.allocate_tensors()
inference_size = input_size(interpreter)

#font options
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (255, 255, 255)

velocidad = "Velocidad 60"

cap = cv2.VideoCapture(0)
state, frame = cap.read()

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter("/home/pi/piCar/DeepPiCar/driver/data/video_result_cam.mp4", fourcc, 30.0, (640,480))

while state:

    state, frame = cap.read()

    preprocessed = img_preprocess(frame)
    X = np.asarray([preprocessed])
    steering_angle = model_road.predict(X)[0]
    steering_angle = steering_angle + 0.5

    final_frame, velocidad = detect_objects(frame,interpreter,inference_size,labels,velocidad)
    image = display_heading_line(final_frame, steering_angle)

    cv2.imshow('Image', image)

    out.write(image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
