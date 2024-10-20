import cv2
import numpy as np
import time
from tqdm import tqdm


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


def countDuplicateFrames(videoPath):
    cap = cv2.VideoCapture(videoPath)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, prevFrame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return

    duplicateCount = 0
    frameCount = 0

    with tqdm(total=100, desc="Processing frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frameCount == 100:  # I only really wanna do 100 frames for testing rn
                break

            frameCount += 1
            result, _ = calcOpticalFlowFarneback(prevFrame, frame)
            if result:
                duplicateCount += 1

            prevFrame = frame
            pbar.update(1)

    cap.release()
    print(f"Total frames: {frameCount}")
    print(f"Duplicate frames: {duplicateCount}")


if __name__ == "__main__":
    # This one should result in a True
    frame1 = cv2.imread(r"G:\TheAnimeScripter\input\frame1.png")
    frame2 = cv2.imread(r"G:\TheAnimeScripter\input\frame2.png")

    startTime = time.time()
    result, value = calcOpticalFlowFarneback(frame1, frame2)
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

    videoPath = r"G:\TheAnimeScripter\input\input.mp4"
    countDuplicateFrames(videoPath)
