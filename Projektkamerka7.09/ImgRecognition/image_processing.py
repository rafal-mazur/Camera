import cv2 as cv
import easyocr as ocr
from matplotlib import pyplot as plt
import numpy as np
import os
import torch

IMGPATH = os.path.join(os.getcwd(), 'imgs', 'notice.jpg')

# return plate position on image
def findPlate(img: cv.Mat):
	pass

# return text form licence plate
def getTextFromImg(img: cv.Mat):
	pass

result = 0
try:
	reader = ocr.Reader(['en'], gpu=False)
except Exception as e:
	print(e)

try:
	result = reader.readtext(IMGPATH)
except Exception as e:
	print(e)

print(result)

print(2)