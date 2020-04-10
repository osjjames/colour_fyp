import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm.auto import tqdm

def compare_frames(frame1, frame2, binCount = 16):
  assert frame1.shape == frame2.shape, "Frames must be of equal dimensions"
  (dimX, dimY) = frame1.shape

  (im1, bins1, patches1) = plt.hist(frame1.ravel(), bins = binCount)  # Create histograms of pixel values
  (im2, bins2, patches2) = plt.hist(frame2.ravel(), bins = binCount)
  plt.clf()

  diff = 0
  for i in range(binCount):
    diff += abs(im1[i] - im2[i])  # Calculate difference between each pair of histogram bins

  similarity = 1 - (diff/(dimX*dimY)) # Similarity as a percentage
  return similarity


def split_video(path, threshold = 0.5, color = False, show_cuts = False, save_to_csv = False): # Returns an array of frame indices, each one is the first frame of a group
  groupStartIndices = [0]
  prevFrame = None
  prevGrayFrame = None
  frameNumber = 0

  vid = cv2.VideoCapture(path)  # Import video

  success,frame = vid.read()   # Read first frame (color order is BGR)
  while success:
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert to grayscale
    if frameNumber != 0:
      similarity = compare_frames(grayFrame, prevGrayFrame)
      if similarity < threshold:
        if show_cuts: # Display each pair of sufficiently different frames
          fig = plt.figure()
          ax1 = fig.add_subplot(1,2,1)
          ax1.imshow(cv2.cvtColor(prevFrame, cv2.COLOR_BGR2RGB) if color else prevGrayFrame)
          ax2 = fig.add_subplot(1,2,2)
          ax2.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if color else grayFrame)
          plt.show()
        print('Similarity: ' + str(similarity))
        groupStartIndices.append(frameNumber)
    prevFrame = frame
    prevGrayFrame = grayFrame
    frameNumber += 1
    success,frame = vid.read()  # Read next frame
  if frameNumber == 0:
    print('Could not load video')
    return None

  if save_to_csv:
    csv_path = os.path.splitext(path)[0] + '.csv'
    np.savetxt(csv_path, np.asarray(groupStartIndices), fmt="%d", delimiter=",")

  return groupStartIndices




