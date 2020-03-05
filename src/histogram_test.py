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


def split_video(path):
  frames = []
  vid = cv2.VideoCapture(path)  # Import video
  success,frame = vid.read()   # Read first frame
  while success:
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert to grayscale
    frames.append(grayFrame)
    success,frame = vid.read()  # Read next frame
  if len(frames) == 0:
    print('Could not load video')

  groups = []
  groups.append([])
  curr_group = 0
  for x in tqdm_notebook(range(len(frames) - 2)):
    similarity = compare_frames(frames[x], frames[x+1])
    groups[curr_group].append(frames[x])

    if similarity < 0.5:
      fig = plt.figure()
      ax1 = fig.add_subplot(1,2,1)
      ax1.imshow(frames[x])
      ax2 = fig.add_subplot(1,2,2)
      ax2.imshow(frames[x+1])
      plt.show()
      print('Similarity: ' + str(similarity))
      groups.append([])
      curr_group += 1

  groups[curr_group].append(frames[len(frames) - 1])
  return groups




