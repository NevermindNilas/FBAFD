import cv2
import numpy as np


def calcOpticalFlowFarneback(frame1, frame2, threshold=1.0, percentage=0.95):
    # Needs to be grayscale, also should be more performant this way
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculates the Optical Flow between two frames
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Applies a threshold to the magnitude of the flow vectors
    zeroFlowCount = np.sum(magnitude < threshold)
    totalVectors = magnitude.size
    zeroFlowPercentage = zeroFlowCount / totalVectors

    # Will return a BOOL value, and the prediction value
    return zeroFlowPercentage > percentage, zeroFlowPercentage
