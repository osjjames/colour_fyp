from shot_cut import split_video

import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm.auto import tqdm, trange
from pathlib import Path

def test_rgb2lab():
  im = cv2.imread('/src/FLIC/images/2-fast-2-furious-00029661.jpg')
  imLab = cv2.cvtColor(im, cv2.COLOR_RGB2Lab)

  fig = plt.figure()
  ax1 = fig.add_subplot(1,3,1)
  ax1.imshow(imLab[:,:,0])
  ax2 = fig.add_subplot(1,3,2)
  ax2.imshow(imLab[:,:,1])
  ax3 = fig.add_subplot(1,3,3)
  ax3.imshow(imLab[:,:,2])
  plt.show()

def training_set_from_video(path, n, use_csv = True):
  groupIndices = split_video(path, show_cuts = True, use_csv = use_csv, save_to_csv = True)

  vid = cv2.VideoCapture(path)  # Import video
  vid_name = Path(path).stem

  train_X = []
  train_y = []

  count = 0;
  for x in trange(len(groupIndices), desc="Group Number"):
    curr_group_Lab = read_group_frames_lab(groupIndices, x)

    color_frame_indices = []
    radius = int(np.floor(n/2)) # every coloured frame should have a radius of n/2 target frames in both directions
    for i in range(radius,len(curr_group_Lab),n): # choose every nth frame to be a coloured frame
      color_frame_indices.append(i)

    for j in tqdm(color_frame_indices, desc="Color Frame Batch"):
      for k in range(j-radius, j+radius+1):
        if k != j and k < len(curr_group_Lab):
          color_a = curr_group_Lab[j][:,:,1]
          color_b = curr_group_Lab[j][:,:,2]
          target = curr_group_Lab[k]
          target_l = curr_group_Lab[k][:,:,0]
          target_ab = curr_group_Lab[k][:,:,1:2]

          X_channels = [target_l, color_a, color_b]
          X_image = np.stack(X_channels, axis=-1)
          name = vid_name + '-' + str(count) + '.png'
          count += 1
          save_lab_image(X_image, '/src/data/train_X/' + name)
          save_lab_image(target, '/src/data/train_y/' + name)

  vid.release()

def save_lab_images(images, folder, name_prefix): # Turn off -ro flag on docker volume for this to work
  for i in range(len(images)):
    save_path = folder + name_prefix + str(i) + '.png'
    save_lab_image(images[i], save_path)

def save_lab_image(image, save_path):
  if not cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_Lab2BGR)):
     raise Exception("Could not write image")

def show_lab_image(image):
  rgb_image = cv2.cvtColor(image, cv2.COLOR_Lab2RGB)
  plt.imshow(rgb_image)

def read_group_frames_lab(group_indices, group_number, vid_capture):
  curr_group_Lab = []
  if group_number != len(group_indices) - 1:
    for y in range(group_indices[group_number+1] - group_indices[group_number]):
      success,frame = vid_capture.read()   # Read next frame
      if success:
        curr_group_Lab.append(cv2.cvtColor(frame, cv2.COLOR_BGR2Lab))
      else:
        print('Video ended before expected')
  else:
    success,frame = vid_capture.read()
    while success: # We don't know how long the last group is, so read frames until we hit the end of the video
      curr_group_Lab.append(cv2.cvtColor(frame, cv2.COLOR_BGR2Lab))
      success,frame = vid_capture.read()

  return curr_group_Lab