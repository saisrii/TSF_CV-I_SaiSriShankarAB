# -*- coding: utf-8 -*-
"""
Created on Sun May  9 21:36:46 2021

@author: Sai_Sri_Shankar
"""


# Importing required the libraries


import numpy as np
import cv2
import pytesseract
import matplotlib.pyplot as plt


# Getting input and plotting of the image

image = cv2.imread('D:/sem 6/DIP/DIP LAB/W1/sai_name.jpg')
plt.imshow(image)

# Converting the image to a grayscale image because it doesnt matter we are only interested in text

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
gray, img_bin = cv2.threshold(gray,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
gray = cv2.bitwise_not(img_bin)


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'



kernel = np.ones((2, 1), np.uint8)

# Dilation is the reverse process with regions growing out from their boundaries.

img = cv2.dilate(image, kernel, iterations=1)

# Erosion involves the removal of pixels ate the edges of the region.


img = cv2.erode(gray, kernel, iterations=1)

# Using the pytesseract to extract the text

out_below = pytesseract.image_to_string(img)
print(out_below)










