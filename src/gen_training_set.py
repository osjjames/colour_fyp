from histogram_test import split_video

import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm_notebook

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

def training_set_from_video(path, n): 
  groups = split_video(path, color = True)

  train_X = []
  train_y = []

  # groups_Lab = [[cv2.cvtColor(frame, cv2.COLOR_RGB2Lab) for frame in groups[i]] for i in range(len(groups))]
  groups_Lab = groups.copy()
  for x in range(len(groups)):
    for y in range(len(groups[x])):
      groups_Lab[x][y] = cv2.cvtColor(groups[x][y], cv2.COLOR_RGB2Lab)

    color_frame_indices = []
    radius = int(np.floor(n/2)) # every coloured frame should have a radius of n/2 target frames in both directions
    for i in range(radius,len(groups_Lab[x]),n): # choose every nth frame to be a coloured frame
      color_frame_indices.append(i)

    for j in tqdm_notebook(color_frame_indices):
      for k in range(j-radius, j+radius+1):
        if k != j and k < len(groups_Lab[x]):
          color_a = groups_Lab[x][j][:,:,1]
          color_b = groups_Lab[x][j][:,:,2]
          target_l = groups_Lab[x][k][:,:,0]
          target_ab = groups_Lab[x][k][:,:,1:2]

          X_channels = [target_l, color_a, color_b]
          X_image = np.stack(X_channels, axis=-1)
          train_X.append(X_image)
          train_y.append(target_ab)

  return [train_X, train_y]

def show_lab_image(image):
  rgb_image = cv2.cvtColor(image, cv2.COLOR_Lab2RGB)
  plt.imshow(rgb_image)



  
 


