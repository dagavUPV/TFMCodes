import cv2
import numpy as np
import math
import sys

import time
import os, os.path
import random

def region_of_interest(edges):
	height, width = edges.shape

	mask = np.zeros(edges.shape).astype(edges.dtype)
	contours = np.array([[
		(500, 1000),
		(750,800),
		(1050, 800),
		(1400,1000)
	]], np.int32)
	cv2.fillPoly(mask, contours, 255)

	cropped_edges = cv2.bitwise_and(edges, mask)
	return cropped_edges

def detect_line_segments(cropped_edges):
	# tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
	rho = 1  # distance precision in pixel, i.e. 1 pixel
	angle = np.pi / 180  # angular precision in radian, i.e. 1 degree
	min_threshold = 25  # minimal of votes
	line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, 
									np.array([]), minLineLength=8, maxLineGap=4)

	return line_segments

def display_lines(frame, lines, line_color=(0, 255, 0), line_width=10):
	line_image = np.zeros_like(frame)
	if lines is not None:
		for line in lines:
			for x1, y1, x2, y2 in line:
				cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
	line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
	return line_image

def make_points(frame, line):
	height, width, _ = frame.shape
	slope, intercept = line
	y1 = height  # bottom of the frame
	y2 = int(y1 * 1 / 2)  # make points from middle of the frame down

	# bound the coordinates within the frame
	x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
	x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
	return [[x1, y1, x2, y2]]

def average_slope_intercept(frame, line_segments):
	"""
	This function combines line segments into one or two lane lines
	If all line slopes are < 0: then we only have detected left lane
	If all line slopes are > 0: then we only have detected right lane
	"""
	lane_lines = []
	#print("----> ",line_segments)
	if line_segments is None:
		return lane_lines

	height, width, _ = frame.shape
	left_fit = []
	right_fit = []

	boundary = 1/3
	left_region_boundary = width * (1 - boundary)  # left lane line segment should be on left 2/3 of the screen
	right_region_boundary = width * boundary # right lane line segment should be on left 2/3 of the screen

	for line_segment in line_segments:
		for x1, y1, x2, y2 in line_segment:
			if x1 == x2:
				continue
			fit = np.polyfit((x1, x2), (y1, y2), 1)
			slope = fit[0]
			intercept = fit[1]
			if slope < 0:
				if x1 < left_region_boundary and x2 < left_region_boundary:
					left_fit.append((slope, intercept))
			else:
				if x1 > right_region_boundary and x2 > right_region_boundary:
					right_fit.append((slope, intercept))

	left_fit_average = np.average(left_fit, axis=0)
	right_fit_average = np.average(right_fit, axis=0)

	#<class 'numpy.ndarray'>
	#print("left_fit_average ->",left_fit_average)
	if len(left_fit) > 0:
		lane_lines.append(make_points(frame, left_fit_average))

	#print("right_fit_average ->",right_fit_average)
	if len(right_fit) > 0:
		lane_lines.append(make_points(frame, right_fit_average))

	return lane_lines

video = sys.argv[1]
out = sys.argv[2]
cap = cv2.VideoCapture(video)

state, frame = cap.read()
count = 0

while state:

	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	#lower_blue = np.array([0, 38, 159])
	lower_blue = np.array([0, 0, 100])
	upper_blue = np.array([255, 255, 255])
	#hsv = cv2.normalize(frame, None, 50, 255, cv2.NORM_MINMAX)

	mask = cv2.inRange(hsv, lower_blue, upper_blue)

	edges = cv2.Canny(mask, 200, 400)

	cropped = region_of_interest(edges)
	line_seg = detect_line_segments(cropped)
	line_seg_img_g1 = display_lines(frame,line_seg)
	line_seg = average_slope_intercept(frame,line_seg)
	line_seg_img_g2 = display_lines(frame,line_seg)

	""" Find the steering angle based on lane line coordinate
		We assume that camera is calibrated to point to dead center
	"""
	lane_lines = line_seg

	if len(lane_lines) == 0:
		print('No lane lines detected, do nothing')
		steering_angle = -90

	else:
		height, width, _ = frame.shape
		if len(lane_lines) == 1:
			print('Only detected one lane line, just follow it. %s' % lane_lines[0])
			x1, _, x2, _ = lane_lines[0][0]
			x_offset = x2 - x1
		else:
			_, _, left_x2, _ = lane_lines[0][0]
			_, _, right_x2, _ = lane_lines[1][0]
			camera_mid_offset_percent = -0.085 # 0.0 means car pointing to center, -0.03: car is centered to left, +0.03 means car pointing to right
			mid = int(width / 2 * (1 + camera_mid_offset_percent))
			x_offset = (left_x2 + right_x2) / 2 - mid

		# find the steering angle, which is angle between navigation direction to end of center line
		y_offset = int(height / 2)

		angle_to_mid_radian = math.atan(x_offset / y_offset)  # angle (in radian) to center vertical line
		angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)  # angle (in degrees) to center vertical line
		steering_angle = angle_to_mid_deg + 90  # this is the steering angle needed by picar front wheel

	cv2.imwrite("{}/video_frame_{}_{}.jpg".format(out,count,steering_angle),frame)
	
	crop_img = frame[750:1000, 500:1450]	#crop image
	mask = np.zeros(crop_img.shape).astype(crop_img.dtype)
	contours = np.array([[
		(0, 0),
		(300,0),
		(0,250)
	]], np.int32)
	crop_img_region = cv2.fillPoly(crop_img, contours, (0,0,0))

	contours = np.array([[
		(700, 0),
		(950,0),
		(950,250)
	]], np.int32)
	crop_img_region = cv2.fillPoly(crop_img_region, contours, (0,0,0))

	cv2.imwrite("{}/crop_video_{}_{}.jpg".format(out,count,steering_angle),crop_img_region)
	print("Frame {}, steering {}.".format(count,steering_angle))
	count += 1
	state, frame = cap.read()

	if cv2.waitKey(1) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
		break

cv2.destroyAllWindows()
