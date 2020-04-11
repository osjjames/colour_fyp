import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm.auto import tqdm

def compare_histograms(hist1, hist2):
  return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


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
      histograms = get_histograms([grayFrame, prevGrayFrame])
      (dimX, dimY) = grayFrame.shape
      similarity = similarity_naive(histograms[0], histograms[1], dimX, dimY)
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
  elif frameNumber < 9:
    print('Video must be longer than 9 frames')
    return None

  if save_to_csv:
    csv_path = os.path.splitext(path)[0] + '.csv'
    np.savetxt(csv_path, np.asarray(groupStartIndices), fmt="%d", delimiter=",")

  return groupStartIndices

def get_histograms(images):
  histograms = []
  for image in images:
    histograms.append(cv2.calcHist([image],[0],None,[256],[0,256]))

  return histograms

def similarity_naive(hist1, hist2, dimX, dimY):
  diff = 0
  for i in range(len(hist1)):
    diff += abs(hist1[i] - hist2[i])  # Calculate difference between each pair of histogram bins

  return 1 - (diff/(dimX*dimY)) # Similarity as a percentage