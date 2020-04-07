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
  groups = split_video(path)

  train_X = []
  train_y = []

  groups_Lab = []
  for x in length(groups):
    for y in length(groups[x]):
      groups_Lab[x][y] = cv2.cvtColor(groups[x][y], cv2.COLOR_RGB2Lab)

    color_frame_indices = []
    radius = np.floor(n/2) # every coloured frame should have a radius of n/2 target frames in both directions
    for i in range(radius,length(groups_Lab[x]),n): # choose every nth frame to be a coloured frame
      color_frame_indices.append(i)

    for j in tqdm_notebook(color_frame_indices):
      for k in range(j-radius, j+radius+1):
        if k != j:
          color_ab = groups_Lab[x][j][:,:,1:2]
          target_l = groups_Lab[x][k][:,:,0]
          target_ab = groups_Lab[x][k][:,:,1:2]

          X_image = []
          X_image[:,:,0] = target_l
          X_image[:,:,1:2] = color_ab
          train_X.append(X_image)
          train_y.append(target_ab)

  return [train_X, train_y]



  
 


