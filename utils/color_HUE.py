#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import sys
import numpy as np
import cv2

img = sys.argv[1]
print(img)

def tmp(x):
	pass

cv2.namedWindow('Bars', cv2.WINDOW_NORMAL)

cv2.createTrackbar('min_blue', 'Bars', 0, 255, tmp)
cv2.createTrackbar('min_green', 'Bars', 0, 255, tmp)
cv2.createTrackbar('min_red', 'Bars', 0, 255, tmp)

cv2.createTrackbar('max_blue', 'Bars', 0, 255, tmp)
cv2.createTrackbar('max_green', 'Bars', 0, 255, tmp)
cv2.createTrackbar('max_red', 'Bars', 0, 255, tmp)

cv2.createTrackbar('alpha', 'Bars', 0, 255, tmp)
cv2.createTrackbar('beta', 'Bars', 255, 255, tmp)

image_BGR = cv2.imread(img)
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.imshow('Image', image_BGR)

image_HSV = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2HSV)
image_HSV_tmp = image_BGR

while True:
	min_blue = cv2.getTrackbarPos('min_blue', 'Bars')
	min_green = cv2.getTrackbarPos('min_green', 'Bars')
	min_red = cv2.getTrackbarPos('min_red', 'Bars')

	max_blue = cv2.getTrackbarPos('max_blue', 'Bars')
	max_green = cv2.getTrackbarPos('max_green', 'Bars')
	max_red = cv2.getTrackbarPos('max_red', 'Bars')

	alpha = cv2.getTrackbarPos('alpha', 'Bars')
	beta = cv2.getTrackbarPos('beta', 'Bars')

	image_HSV_tmp = cv2.normalize(image_BGR, None, alpha, beta, cv2.NORM_MINMAX)

	cv2.namedWindow('Brightness', cv2.WINDOW_NORMAL)
	cv2.imshow('Brightness', image_HSV_tmp)

	mask = cv2.inRange(image_HSV_tmp,
					   (min_blue, min_green, min_red),
					   (max_blue, max_green, max_red))

	cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
	cv2.imshow('Mask', mask)

	edges = cv2.Canny(mask, 200, 400)
	cv2.namedWindow('Edges', cv2.WINDOW_NORMAL)
	cv2.imshow('Edges', edges)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break


cv2.destroyAllWindows()


print('min_blue, min_green, min_red = {0}, {1}, {2}'.format(min_blue, min_green, min_red))
print('max_blue, max_green, max_red = {0}, {1}, {2}'.format(max_blue, max_green, max_red))
print('alpha, beta = {0}, {1}'.format(alpha,beta))
