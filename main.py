import cv2
import numpy as np
import time


def areFramesDuplicates(frame1, frame2, threshold=1.0, percentage=0.95):
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


if __name__ == "__main__":
    # This one should result in a True
    frame1 = cv2.imread(r"G:\TheAnimeScripter\input\frame1.png")
    frame2 = cv2.imread(r"G:\TheAnimeScripter\input\frame2.png")

    startTime = time.time()
    result, value = areFramesDuplicates(frame1, frame2)
    endTime = time.time()
    print(f"Prediction Value: {value}")

    elapsedTime = endTime - startTime
    fps = 1 / elapsedTime if elapsedTime > 0 else float("inf")

    if result:
        print("Frames are duplicates")
    else:
        print("Frames are not duplicates")

    print(f"Time taken to compare frames: {elapsedTime:.6f} seconds")
    print(f"Estimated FPS: {fps:.2f}")
