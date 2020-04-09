import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm_notebook

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


def split_video(path, color = False):
  colorFrames = []
  grayFrames = []
  vid = cv2.VideoCapture(path)  # Import video
  success,frame = vid.read()   # Read first frame (color order is BGR)
  while success:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    colorFrames.append(frame)
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # Convert to grayscale
    grayFrames.append(grayFrame)
    success,frame = vid.read()  # Read next frame
  if len(grayFrames) == 0:
    print('Could not load video')

  groups = []
  groups.append([])
  curr_group = 0
  for x in tqdm_notebook(range(len(grayFrames) - 2)):
    similarity = compare_frames(grayFrames[x], grayFrames[x+1])
    groups[curr_group].append(colorFrames[x] if color else grayFrames[x])

    if similarity < 0.5:
      fig = plt.figure()
      ax1 = fig.add_subplot(1,2,1)
      ax1.imshow(colorFrames[x] if color else grayFrames[x])
      ax2 = fig.add_subplot(1,2,2)
      ax2.imshow(colorFrames[x+1] if color else grayFrames[x+1])
      plt.show()
      print('Similarity: ' + str(similarity))
      groups.append([])
      curr_group += 1

  groups[curr_group].append(colorFrames[len(colorFrames) - 1] if color else grayFrames[len(grayFrames) - 1])
  return groups




