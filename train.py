# train.py
# Given training images, Produces training data and classification files for A-Z, 0-9 characters.
# Contributors: Kyla (main), Edrienne

import os
import sys
import numpy as np
import cv2
import contour_helper as help
np.set_printoptions(threshold=np.inf)

# Constants
MIN_CONTOUR_AREA = 50
RESIZED_IMAGE_WIDTH = 40
RESIZED_IMAGE_HEIGHT = 60

training_classification_filename = 'training_classification_labels'
testing_classification_filename = 'testing_classification_labels'
training_data_filename = 'training_data'
testing_data_filename = 'testing_data'
HAND_DATA_DIR = "English/Hnd/Img/Sample0"
FONT_DATA_DIR = "English/Fnt/Sample0"

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
	charX, charY = trainingImg.shape[1::-1]
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
	np.savetxt(trainingDataFile, trainingdata.astype(int), fmt='%i')
	trainingDataFile.close()
	# remove windows from memory
	cv2.destroyAllWindows() 

	return

def processImages(directoryPath, character_class, trainNum, trainingClassificationArray, testingClassificationArray):
	for idx, file in enumerate(os.listdir(os.path.expanduser(directoryPath))):
		classificationArray = trainingClassificationArray
		destinationFile = training_data_filename
		if (idx > 500):
			return
		if (idx > trainNum):
			classificationArray = testingClassificationArray
			destinationFile = testing_data_filename

		filePath = os.path.join(directoryPath, file)
		classificationArray.append(character_class)
		extractFeatures(filePath, destinationFile)


def main():
	""" Classifies training data images with uppercase and number labels """
	global training_data_filename
	global testing_data_filename
	global training_classification_filename
	global testing_classification_filename

	sampleRange = range(1, 63)

	if (sys.argv[1] == "d"):
		training_data_filename += "_digit"
		testing_data_filename += "_digit"
		training_classification_filename += "_digit"
		testing_classification_filename += "_digit"
		sampleRange = range(1, 11)
	elif (sys.argv[1] == "u"):
		training_data_filename += "_uppercase"
		testing_data_filename += "_uppercase"
		training_classification_filename += "_uppercase"
		testing_classification_filename += "_uppercase"
		sampleRange = range(11, 37)
	elif (sys.argv[1] == "l"):
		training_data_filename += "_lowercase"
		testing_data_filename += "_lowercase"
		training_classification_filename += "_lowercase"
		testing_classification_filename += "_lowercase"
		sampleRange = range(37, 63)
	elif (sys.argv[1] == "a"):
		training_data_filename += "_alphabet"
		testing_data_filename += "_alphabet"
		training_classification_filename += "_alphabet"
		testing_classification_filename += "_alphabet"
		sampleRange = range(11, 63)

	if (sys.argv[2] == "h-h"):
		training_data_filename += "_h-h"
		testing_data_filename += "_h-h"
		training_classification_filename += "_h-h"
		testing_classification_filename += "_h-h"
	elif (sys.argv[2] == "f-f"):
		training_data_filename += "_f-f"
		testing_data_filename += "_f-f"
		training_classification_filename += "_f-f"
		testing_classification_filename += "_f-f"
	elif (sys.argv[2] == "fh-h"):
		training_data_filename += "_fh-h"
		testing_data_filename += "_fh-h"
		training_classification_filename += "_fh-h"
		testing_classification_filename += "_fh-h"
	elif (sys.argv[2] == "f-h"):
		training_data_filename += "_f-h"
		testing_data_filename += "_f-h"
		training_classification_filename += "_f-h"
		testing_classification_filename += "_f-h"
	

	training_data_filename += ".txt"
	testing_data_filename += ".txt"
	training_classification_filename += ".txt"
	testing_classification_filename += ".txt"

	open(training_classification_filename, 'w').close()
	open(testing_classification_filename, 'w').close()
	open(training_data_filename, 'w').close()
	open(testing_data_filename, 'w').close()

	trainingClassificationArray = []
	testingClassificationArray = []

	for i in sampleRange:
		character_class = str(i).zfill(2)

		if (sys.argv[2] == "h-h"):
			handDirectory = os.path.expanduser(HAND_DATA_DIR + character_class)
			processImages(handDirectory, character_class, 40, trainingClassificationArray, testingClassificationArray)
		elif (sys.argv[2] == "f-f"):
			fontDirectory = os.path.expanduser(FONT_DATA_DIR + character_class)
			processImages(fontDirectory, character_class, 381, trainingClassificationArray, testingClassificationArray)
		elif (sys.argv[2] == "fh-h"):
			handDirectory = os.path.expanduser(HAND_DATA_DIR + character_class)
			processImages(handDirectory, character_class, 40, trainingClassificationArray, testingClassificationArray)
			fontDirectory = os.path.expanduser(FONT_DATA_DIR + character_class)
			processImages(fontDirectory, character_class, 500, trainingClassificationArray, testingClassificationArray)
		elif (sys.argv[2] == "f-h"):
			handDirectory = os.path.expanduser(HAND_DATA_DIR + character_class)
			processImages(handDirectory, character_class, 0, trainingClassificationArray, testingClassificationArray)
			fontDirectory = os.path.expanduser(FONT_DATA_DIR + character_class)
			processImages(fontDirectory, character_class, 500, trainingClassificationArray, testingClassificationArray)
		
		print i
	
	floatClassificationsTrain = np.array(trainingClassificationArray, np.float32)
	floatClassificationsTest = np.array(testingClassificationArray, np.float32)

	# flatten to 1d
	flatClassificationsTrain = floatClassificationsTrain.reshape((floatClassificationsTrain.size, 1))
	flatClassificationsTest = floatClassificationsTest.reshape((floatClassificationsTest.size, 1))

	classificationFileTrain = open(training_classification_filename, 'a')
	classificationFileTest = open(testing_classification_filename, 'a')

	np.savetxt(classificationFileTrain, flatClassificationsTrain)
	np.savetxt(classificationFileTest, flatClassificationsTest)
	classificationFileTrain.close()
	classificationFileTest.close()

	print "Training & testing data generated"
	return
	
if __name__ == "__main__":
	main()