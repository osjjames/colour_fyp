import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

def compare_frames(frame1, frame2, binCount = 16):
  assert frame1.shape == frame2.shape, "Frames must be of equal dimensions"
  (dimX, dimY) = frame1.shape

  (im1, bins1, patches1) = plt.hist(frame1.ravel(), bins = binCount)  # Create histograms of pixel values
  (im2, bins2, patches2) = plt.hist(frame2.ravel(), bins = binCount)
  plt.clf()

  diff = 0
  for i in range(binCount):
    diff += abs(im1[i] - im2[i])  # Calculate difference between each pair of historgram bins

  similarity = 1 - (diff/(dimX*dimY)) # Similarity as a percentage
  return similarity



frames = []
vid = cv2.VideoCapture('/src/vid5.mp4')  # Import video
success,frame = vid.read()   # Read first frame
while success:
  grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert to grayscale
  frames.append(grayFrame)
  success,frame = vid.read()  # Read next frame

for x in tqdm(range(len(frames) - 1)):
  similarity = compare_frames(frames[x], frames[x+1])

  if similarity < 0.6:
    plt.imshow(frames[x])
    plt.show()
    plt.imshow(frames[x+1])
    plt.show()
    print(similarity)



