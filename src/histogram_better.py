import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm.auto import tqdm

# Using HCSM9 from http://www.ee.surrey.ac.uk/CVSSP/Publications-/papers/yusof-bmvc2000.pdf

def compare_histograms(hist1, hist2):
  return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)

def split_video(path, show_cuts = False, use_csv = True, save_to_csv = True): # Returns an array of frame indices, each one is the first frame of a group
  if use_csv:
    group_start_indices = check_for_csv(path)
    if group_start_indices != None:
      return group_start_indices

  group_start_indices = [0]
  window_size = 9
  window_centre = window_size // 2
  frame_number = 0

  window = [] # Stores consecutive frames, and slides along 1 frame each iteration. Length is always window_size
  window_histograms = [] # Stores histograms of corresponding frames
  window_samples = [] # Stores dissimilarity values between window[k] and window[k+1] (called the sample of window[k])
  next_frame = []
  next_histogram = []
  next_decision = 0

  vid = cv2.VideoCapture(path)  # Import video
  success,frame = vid.read() # Read first frame (color order is BGR)
  for i in range(window_size): # Gather frames of initial window
    if not success: break
    window.append(frame)
    window_histograms.append(get_histogram(frame)) # Calculate histogram of each frame
    if i > 0:
      window_samples.append(compare_histograms(window_histograms[-2], window_histograms[-1])) # Compute sample of frame as soon as next frame has been loaded
    frame_number += 1
    success,frame = vid.read()

  while success:
    next_frame = frame # Load up next frame (not included in the current window but needed for sample computation)
    next_histogram = get_histogram(next_frame)
    window_samples.append(compare_histograms(window_histograms[-1], next_histogram)) # Compute the last sample (this is why we loaded up the next frame)
    threshold = get_threshold(window_samples)
    dissimilarity = window_samples[window_centre]
    
    if dissimilarity > threshold and frame_number > next_decision:
      if show_cuts: # Display each pair of sufficiently different frames
        display_frame_pair(window[window_centre], window[window_centre+1])
      print('Dissimilarity: ' + str(dissimilarity))
      group_start_indices.append(frame_number-window_centre)
      next_decision = frame_number + window_centre # After a shot is detected, no new decisions are made until half the window size has passed

    window.pop(0) # Shift all window elements to the left to make room for the data of the next frame
    window.append(next_frame)
    window_histograms.pop(0)
    window_histograms.append(next_histogram)
    window_samples.pop(0) # Remove the first element, but we cannot append the new sample until the next frame is loaded up

    frame_number += 1
    success,frame = vid.read()  # Read next frame
  if frame_number == 0:
    print('Could not load video')
    return None
  elif frame_number < window_size:
    print('Video must be longer than ' + str(window_size) + ' frames')
    return None

  if save_to_csv:
    csv_path = os.path.splitext(path)[0] + '.csv'
    np.savetxt(csv_path, np.asarray(group_start_indices), fmt="%d", delimiter=",")

  vid.release()
  return group_start_indices

def check_for_csv(path):
  csv_path = os.path.splitext(path)[0] + '.csv'
  if os.path.isfile(csv_path):
    with open(csv_path, newline='') as file: # Read csv file of frame group indices
      reader = csv.reader(file)
      data = list(reader)
    data = np.ravel(data) # Flatten the list
    data = [int(i) for i in data] # Convert all elements to ints
    return data
  else:
    return None

def get_histogram(image):
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert to grayscale
  return cv2.calcHist([gray_image],[0],None,[256],[0,256])

def get_threshold(window_samples):
  mean = sum(window_samples) / len(window_samples)
  return mean*7 # Second value is a threshold in itself: how many times more dissimilar than the mean must a pair of frames be to count as a cut

def display_frame_pair(frame1, frame2):
  plt.clf()
  fig = plt.figure()
  ax1 = fig.add_subplot(1,2,1)
  ax1.imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
  ax2 = fig.add_subplot(1,2,2)
  ax2.imshow(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
  plt.show()