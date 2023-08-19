import cv2
import numpy as np


image = cv2.imread('/Users/muhammadhamzasohail/Desktop/Digital-Ammeter-CEAR-WS±150A.jpeg')
image = cv2.resize(image, None, fx=0.9, fy=0.9)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
ret, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Detect the contours
contours, hierarchy = cv2.findContours(binary, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

min_contour_area = 8000  #
max_contour_area = 55000
larger_contours = [cnt for cnt in contours if  min_contour_area < cv2.contourArea(cnt) < max_contour_area]

# draw contours 
print(len(contours))
image_copy = image.copy()
image_copy = cv2.drawContours(image_copy, larger_contours, -1, (0, 255, 0), thickness=2)

# Visualize the results 
cv2.imshow('Grayscale Image', gray)
cv2.imshow('Drawn Contours', image_copy)
cv2.imshow('Binary Image', binary)

cv2.waitKey(0)
cv2.destroyAllWindows()
