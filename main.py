import cv2
from tqdm import tqdm
from src.cv2FlowFarneback import calcOpticalFlowFarneback
from src.pytorchFlowRaft import pytorchFlowRaft
from skimage.metrics import structural_similarity as ssim


def countDuplicateFrames(videoPath, useTorch=False, precision="fp32"):
    cap = cv2.VideoCapture(videoPath)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, prevFrame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return

    frameCount = 0
    duplicateCount = 0

    with tqdm(total=100, desc="Processing frames") as pbar:
        while frameCount < 100:
            ret, frame = cap.read()
            if not ret:
                break

            if useTorch:
                if precision == "fp16":
                    result, _ = pytorchFlowRaft(prevFrame, frame, precision="fp16")
                else:
                    result, _ = pytorchFlowRaft(prevFrame, frame, precision="fp32")
            else:
                result, _ = calcOpticalFlowFarneback(prevFrame, frame)
            if result:
                duplicateCount += 1

            frameCount += 1
            prevFrame = frame
            pbar.update(1)

    cap.release()
    print(f"Duplicate frames: {duplicateCount}")


def getBaseline(videoPath):
    cap = cv2.VideoCapture(videoPath)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    ret, prevFrame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return

    frameCount = 0
    duplicateCount = 0

    prevFrameGray = cv2.cvtColor(prevFrame, cv2.COLOR_BGR2GRAY)

    with tqdm(total=100, desc="Processing frames") as pbar:
        while frameCount < 100:
            ret, frame = cap.read()
            if not ret:
                break

            frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            score, _ = ssim(prevFrameGray, frameGray, full=True)

            if (
                score > 0.995
            ):  # Should be relatively close to 1, hard to know which value to use
                duplicateCount += 1

            prevFrameGray = frameGray
            frameCount += 1
            pbar.update(1)

    cap.release()
    print(f"Duplicate: {duplicateCount}")


if __name__ == "__main__":
    videoPath = r"C:\Users\nilas\Downloads\test10.mp4"

    print("Getting baseline with SSIM 0.995...")
    getBaseline(videoPath)
    print("Running OpenCV implementation...")
    countDuplicateFrames(videoPath, useTorch=False)
    print("Running PyTorch FP32 implementation...")
    countDuplicateFrames(videoPath, useTorch=True, precision="fp32")
    # print("Running PyTorch FP16 implementation...")
    # countDuplicateFrames(videoPath, useTorch=True, precision="fp16")
