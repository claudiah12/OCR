# train.py
# Given training images, Produces training data and classification files for A-Z, 0-9 characters.
# Contributors: Kyla (main), Edrienne

import os
import numpy as np
import cv2
import contour_helper as help
np.set_printoptions(threshold=np.inf)

# Constants
MIN_CONTOUR_AREA = 100
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

CLASSIFICATION_FILENAME = 'classification_labels.txt'
TRAINING_DATA_FILENAME = 'training_data.txt'
TRAIN_DATA_DIR = "English/Hnd/Img/Sample0"

# Flags
showImages = False # whether to cv2.imshow() the results
showContourOrder = False # whether to show order of contours being classified
checkForTittles = False # Keep False until lowercase letters trained

def classifyImage(trainingImageName):
	"""
	Produce training data and classification files given filename and classification array
	:param trainingImageName:
	:param classificationArray:
	"""
	# open or create classification and training data files
	trainingDataFile = file(TRAINING_DATA_FILENAME, 'a')
	
	# read in training image and apply preprocessing functions
	trainingImg = cv2.imread(trainingImageName)
	grayImg = cv2.cvtColor(trainingImg, cv2.COLOR_BGR2GRAY)
	blurImg = cv2.blur(grayImg,(5,5))
	threshImg = cv2.adaptiveThreshold(blurImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

	# find and sort contours
	contours, hierarchy = cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours.sort(key=lambda x: help.sortContoursUpperLeftToLowerRight(x))
	
	# declare empty array with size equal to number of training data samples
	trainingdata = np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT), dtype=int)
	
	# find contours that have a tittle on top (i's or j's)
	if checkForTittles:
		lettersWithTittles = []
		tittles = []
		for i in range(len(contours)-1):
			if help.detectTittles(contours[i], contours[i+1]):
				lettersWithTittles.append(contours[i])
				tittles.append(contours[i+1])

	# add appropriate contours to training data
	for contour in contours:
			if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
				# get bounding rect for current contour
				[intX, intY, intW, intH] = cv2.boundingRect(contour)

				if checkForTittles and help.getIndexOfTittle(contour, lettersWithTittles) > -1:
					index = help.getIndexOfTittle(contour, lettersWithTittles)
					if index > -1:
						# get dimensions of tittle and use to draw rect and grab letter
						[tX, tY,tWidth, tHeight] = cv2.boundingRect(tittles[index])
						additionalHeight = intY - (tY + tHeight)

						cv2.rectangle(trainingImg,(intX, tY),(intX + intW, tY + intH + tHeight + additionalHeight),(255, 0, 0),1)
						contourImg = threshImg[intY:intY + intH + tHeight + additionalHeight, intX:intX + intW]
				else:
					# draw rect and grab letter
					cv2.rectangle(trainingImg, (intX, intY), (intX + intW, intY + intH), (255, 0, 255), 1)
					contourImg = threshImg[intY:intY + intH, intX:intX + intW]
				
				# resize image and show on original
				contourImgResized = cv2.resize(contourImg, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
				if showImages:
					cv2.imshow(trainingImageName + " thresh		olded", threshImg)
					cv2.imshow(trainingImageName, trainingImg)
				
				# flatten contour to 1D and append to training data
				contourImgFlatten = contourImgResized.reshape((1,RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
				trainingdata = np.append(trainingdata, contourImgFlatten,0)
				
				# show order that contours are classified
				if showImages and showContourOrder:
					cv2.imshow(trainingImageName, trainingImg)
					cv2.waitKey(0)
		
	np.savetxt(trainingDataFile, trainingdata)
	trainingDataFile.close()
	# remove windows from memory
	cv2.destroyAllWindows() 

	return


def main():
	""" Classifies training data images with uppercase and number labels """

	open(CLASSIFICATION_FILENAME, 'w').close()
	open(TRAINING_DATA_FILENAME, 'w').close()

	classificationArray = []

	for i in range(1, 2):
		character_class = str(i).zfill(2)
		directoryPath = os.path.expanduser(TRAIN_DATA_DIR + character_class)
		for file in os.listdir(os.path.expanduser(directoryPath)):
				
			filePath = os.path.join(directoryPath, file)
			classificationArray.append(character_class)
			classifyImage(filePath)
		
	floatClassifications = np.array(classificationArray, np.float32)
	# flatten to 1d
	flatClassifications = floatClassifications.reshape((floatClassifications.size, 1))
	classificationFile = open(CLASSIFICATION_FILENAME, 'a')
	np.savetxt(classificationFile, flatClassifications)
	classificationFile.close()

	print "Training data generated"
	return
	
if __name__ == "__main__":
	main()