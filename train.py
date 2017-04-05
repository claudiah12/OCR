# train.py
# Given training images, Produces training data and classification files for A-Z, 0-9 characters.
# Contributors: Kyla (main), Edrienne

import os
import numpy as np
import cv2
import contour_helper as help
np.set_printoptions(threshold=np.inf)

# Constants
MIN_CONTOUR_AREA = 1000
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

TRAINING_CLASSIFICATION_FILENAME = 'training_classification_labels.txt'
TESTING_CLASSIFICATION_FILENAME = 'testing_classification_labels.txt'
TRAINING_DATA_FILENAME = 'training_data.txt'
TESTING_DATA_FILENAME = 'testing_data.txt'
TRAIN_DATA_DIR = "English/Hnd/Img/Sample0"

# Flags
showImages = False # whether to cv2.imshow() the results
showContourOrder = False # whether to show order of contours being classified
checkForTittles = False # Keep False until lowercase letters trained

def extractFeatures(trainingImageName, destinationFile):
	"""
	Produce training data and classification files given filename and classification array
	:param trainingImageName:
	:param classificationArray:
	"""
	# open or create classification and training data files
	trainingDataFile = file(destinationFile, 'a')
	
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
	charX, charY = trainingImg.shape[1::-1];
	charW = 0
	charH = 0
	# add appropriate contours to training data
	for contour in contours:
			if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
				# get bounding rect for current contour
				[intX, intY, intW, intH] = cv2.boundingRect(contour)
				charX = min(charX, intX)
				charY = min(charY, intY)
				charW = max(charW, intW)
				charH = max(charH, intH)

				# if checkForTittles and help.getIndexOfTittle(contour, lettersWithTittles) > -1:
				# 	index = help.getIndexOfTittle(contour, lettersWithTittles)
				# 	if index > -1:
				# 		# get dimensions of tittle and use to draw rect and grab letter
				# 		[tX, tY,tWidth, tHeight] = cv2.boundingRect(tittles[index])
				# 		additionalHeight = intY - (tY + tHeight)

				# 		cv2.rectangle(trainingImg,(intX, tY),(intX + intW, tY + intH + tHeight + additionalHeight),(255, 0, 0),1)
				# 		contourImg = threshImg[intY:intY + intH + tHeight + additionalHeight, intX:intX + intW]
				# else:
				# 	# draw rect and grab letter
	cv2.rectangle(trainingImg, (charX, charY), (charX + charW, charY + charH), (255, 0, 255), 1)
	contourImg = threshImg[charY:charY + charH, charX:charX + charW]
	
	# resize image and show on original
	contourImgResized = cv2.resize(contourImg, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
	
	if showImages:
		print trainingImageName
		cv2.imshow(trainingImageName + " thresholded", threshImg)
		cv2.waitKey(0)
		#cv2.imshow(trainingImageName, trainingImg)

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

	if(destinationFile == TRAINING_DATA_FILENAME):
		return len(trainingdata)
	return 0


def main():
	""" Classifies training data images with uppercase and number labels """

	open(TRAINING_CLASSIFICATION_FILENAME, 'w').close()
	open(TESTING_CLASSIFICATION_FILENAME, 'w').close()
	open(TRAINING_DATA_FILENAME, 'w').close()
	open(TESTING_DATA_FILENAME, 'w').close()

	trainingClassificationArray = []
	testingClassificationArray = []
	count = 0

	trainingImg = cv2.imread("English/Hnd/Img/Sample005/img005-041.png")

	for i in range(1, 10):
		character_class = str(i).zfill(2)
		directoryPath = os.path.expanduser(TRAIN_DATA_DIR + character_class)
		for idx, file in enumerate(os.listdir(os.path.expanduser(directoryPath))):
			classificationArray = trainingClassificationArray
			destinationFile = TRAINING_DATA_FILENAME
			if (idx > 40):
				classificationArray = testingClassificationArray
				destinationFile = TESTING_DATA_FILENAME

			filePath = os.path.join(directoryPath, file)
			classificationArray.append(character_class)
			count += extractFeatures(filePath, destinationFile)
		print i
	
	floatClassificationsTrain = np.array(trainingClassificationArray, np.float32)
	floatClassificationsTest = np.array(testingClassificationArray, np.float32)

	# flatten to 1d
	flatClassificationsTrain = floatClassificationsTrain.reshape((floatClassificationsTrain.size, 1))
	flatClassificationsTest = floatClassificationsTest.reshape((floatClassificationsTest.size, 1))

	classificationFileTrain = open(TRAINING_CLASSIFICATION_FILENAME, 'a')
	classificationFileTest = open(TESTING_CLASSIFICATION_FILENAME, 'a')

	np.savetxt(classificationFileTrain, flatClassificationsTrain)
	np.savetxt(classificationFileTest, flatClassificationsTest)
	classificationFileTrain.close()
	classificationFileTest.close()

	print "Training & testing data generated"
	return
	
if __name__ == "__main__":
	main()