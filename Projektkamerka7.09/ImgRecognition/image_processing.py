import cv2 as cv
import easyocr as ocr
from matplotlib import pyplot as plt
import numpy as np
import os

IMGPATH = os.path.join(os.getcwd(), 'imgs', 'notice.jpg')

# return plate position on image
def findPlate(img: cv.Mat):
	pass

# return text form licence plate
def getTextFromImg(img: cv.Mat):
	pass

reader = ocr.Reader(['en'], gpu=False)
result = reader.readtext(IMGPATH)

print(result)
