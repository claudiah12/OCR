# knn.py

import sys
import cv2
import numpy as np
import contour_helper as help

import train

# Character Contour Criteria
MIN_WIDTH = 2
MIN_HEIGHT = 5
MIN_CONTOUR_AREA = 100
MIN_ASPECT_RATIO = 0.15
MAX_ASPECT_RATIO = 3.9

# Flags
showImages = True
showContourOrder = True

def __getTrainedKNearest(training_filename, training_classification_filename):
	"""
	Runs K-Nearest Neighbors classification using classification and training data
	:return: kNearest object
	"""
	try:
		classificationLabels = np.loadtxt(training_classification_filename, np.float32)
	except:
		print("Can't find classification labels")
	
	try:
		trainingData = np.loadtxt(training_filename, np.float32)
	except:
		print("Can't find training data")

	classificationLabels = classificationLabels.reshape((classificationLabels.size, 1))  # reshape to 1D for train()
	print "class", classificationLabels.shape
	print "train", trainingData.shape
	# create KNearest and train
	kNearest = cv2.KNearest()
	kNearest.train(trainingData, classificationLabels)

	return kNearest	

def classify(letter, training_filename, training_classification_filename):
	kNearest = __getTrainedKNearest(training_filename, training_classification_filename)
	# flatten and convert to numpy array of floats

	# find k nearest neighbor to determine character
	ret, result, neighbors, dist = kNearest.find_nearest(letter, 2)
	print "r", ret
	return result

def main():
	training_classification_filename = 'training_classification_labels'
	testing_classification_filename = 'testing_classification_labels'
	training_data_filename = 'training_data'
	testing_data_filename = 'testing_data'

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

	training_data_filename += ".txt"
	testing_data_filename += ".txt"
	training_classification_filename += ".txt"
	testing_classification_filename += ".txt"

	testing_data = np.loadtxt(testing_data_filename)
	testing_classifications = np.loadtxt(testing_classification_filename)
	testing_data = np.float32(testing_data)

	kNearest = __getTrainedKNearest(training_data_filename, training_classification_filename)
	ret, result, neighbors, dist = kNearest.find_nearest(testing_data, 5)
	correct = 0
	for idx, r in enumerate(result):
		#print r[0], "=", testing_classifications[idx]
		if r[0] == testing_classifications[idx]:
			correct += 1
	print correct / float(len(result))
	#test_results =np.apply_along_axis( classify, 1, testing_data )
	

if __name__ == "__main__":
	main()