# contour_helper.py
# Helper functions for dealing with contours

import cv2
import math
import numpy as np


def sortContoursUpperLeftToLowerRight(contour):
    """
	Method used to sort contours from left to right, top to bottom
	:param contour:
	:return: Score used to sort contours
	"""
    mom1 = cv2.moments(contour)
    x, y = 0, 0 # centroid
    if mom1["m00"] != 0:
        x = int(mom1["m10"] / mom1["m00"])
        y = int(mom1["m01"] / mom1["m00"])

    tolerance_factor = 100
    value = value = ((y // tolerance_factor) * tolerance_factor) * 100 + x

    return value

def detectTittles(contour1, contour2):
    """
    :param contour1:
    :param contour2:
    :return: Whether two contours are a character with a tittle
    """
    tolerance_factor = 5
    mom1 = cv2.moments(contour1)
    mom2 = cv2.moments(contour2)

    x1, x2 = 0, 0
    if mom1["m00"] != 0:
        x1 = int(mom1["m10"] / mom1["m00"])
        x2 = int(mom1["m01"] / mom1["m00"])

    if (abs(x1 - x2) < tolerance_factor):
        return True
    else:
        return False

def getDistanceBetween(contour1, contour2):
    """
    :param contour1:
    :param contour2:
    :return: Distance between the center of charA and charB using the Pythagorean theorem
    """
    momContour1 = cv2.moments(contour1)
    momContour2 = cv2.moments(contour2)

    cXA, cYA, cXB, cYB = 0, 0, 0, 0

    if (momContour1["m00"] != 0):
        cXA = int(momContour1["m10"] / momContour1["m00"])
        cYA = int(momContour1["m01"] / momContour1["m00"])

    if (momContour2["m00"] != 0):
        cXB = int(momContour2["m10"] / momContour2["m00"])
        cYB = int(momContour2["m01"] / momContour2["m00"])

    intX = abs(cXA - cXB)
    intY = abs(cYA - cYB)

    return math.sqrt((intX ** 2) + (intY ** 2))


def getMeanDistanceBetweenContours(contourList):
    """
    :param contourList:
    :return: mean, stdDev of the distances of the contours
    """
    # get mean distance between contours
    distanceList = []
    for i in range(len(contourList) - 1):
        contour = contourList[i]
        nextChar = contourList[i + 1]
        distance = getDistanceBetween(nextChar, contour)
        distanceList.append(distance)

    mean, stdDev = 0, 0
    if (len(distanceList) > 0):
        mean = np.mean(distanceList)
        stdDev = np.std(distanceList)

    return mean, stdDev


def getIndexOfTittle(contour, lettersWithTittles):
    """
    :param contour:
    :param lettersWithTittles:
    :return: i The index of the first contour that has a letter with a tittle
    """
    for i in range(len(lettersWithTittles)):
        if (np.any(contour == lettersWithTittles[i])):
            return i
    return -1