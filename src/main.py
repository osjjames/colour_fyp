import time
from tqdm.auto import tqdm, trange
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

import cnn
import zhang
import gen_training_set as gts
import histogram_better as hb
import video_utils as vu

def colorise_video(video_path, n):
  save_path = os.path.splitext(video_path)[0] + '-colourised-' + str(n) + '.mp4'
  vid_out = vu.setup_writer(video_path, save_path)

  model = cnn.load_model(cnn.checkpoint_models_path + "full_model_256.hdf5") # Load CNN model
  print('model loaded')

  group_indices = hb.split_video(video_path) # Split input video by shots

  vid_in = cv2.VideoCapture(video_path)  # Import video

  for group_number in trange(len(group_indices), desc="Group Number"):
    curr_group_Lab = gts.read_group_frames_lab(group_indices, group_number, vid_in)

    color_frame_indices = []
    radius = int(np.floor(n/2)) # every coloured frame should have a radius of n/2 target frames in both directions
    for i in range(radius,len(curr_group_Lab),n): # choose every nth frame to be a coloured frame
      color_frame_indices.append(i)

    for j in tqdm(color_frame_indices, desc="Color Frame Batch"):
      colour_frame = zhang.colorize(curr_group_Lab[j], lab_only = True) # Colourise every nth frame
      # colour_frame = curr_group_Lab[j]
      hb.display_frame_pair(cv2.cvtColor(curr_group_Lab[j], cv2.COLOR_Lab2BGR), cv2.cvtColor(colour_frame, cv2.COLOR_Lab2BGR))
      color_a = colour_frame[:,:,1] # Extract a + b channels
      color_b = colour_frame[:,:,2]
      for k in range(j-radius, j+radius+1):
        if k != j and k < len(curr_group_Lab):
          target = curr_group_Lab[k]
          target_l = target[:,:,0]

          X_channels = [target_l, color_a, color_b]
          X_image = np.stack(X_channels, axis=-1) # Combine target frame L channel with colourised frame a+b channels

          new_frame = cnn.predict_lab(model, X_image) # Put through CNN to smooth colour differences
          hb.display_frame_pair(cv2.cvtColor(X_image, cv2.COLOR_Lab2BGR), cv2.cvtColor(new_frame, cv2.COLOR_Lab2BGR))
          new_frame_rgb = cv2.cvtColor(new_frame, cv2.COLOR_Lab2BGR)

        if k == j and k < len(curr_group_Lab):
          new_frame_rgb = cv2.cvtColor(colour_frame, cv2.COLOR_Lab2BGR)

        vid_out.write(new_frame_rgb) # Save frame to video

  vid_in.release()
  vid_out.release()

def colorise_video_one(video_path):
  save_path = os.path.splitext(video_path)[0] + '-colourised-one.mp4'
  vid_out = vu.setup_writer(video_path, save_path)

  model = cnn.load_model(cnn.checkpoint_models_path + "full_model_256.hdf5") # Load CNN model
  print('model loaded')

  group_indices = hb.split_video(video_path) # Split input video by shots

  vid_in = cv2.VideoCapture(video_path)  # Import video

  for group_number in trange(len(group_indices), desc="Group Number"):
    curr_group_Lab = gts.read_group_frames_lab(group_indices, group_number, vid_in)

    # colour_frame = zhang.colorize(curr_group_Lab[j], lab_only = True) # Colourise every nth frame
    colour_frame = curr_group_Lab[0]
    vid_out.write(cv2.cvtColor(colour_frame, cv2.COLOR_Lab2BGR)) # Save frame to video
    
    prev_frame = colour_frame
    for j in trange(1, len(curr_group_Lab), desc="Frame number within group"):
      color_a = prev_frame[:,:,1] # Extract a + b channels
      color_b = prev_frame[:,:,2]
      target = curr_group_Lab[j]
      target_l = target[:,:,0]

      X_channels = [target_l, color_a, color_b]
      X_image = np.stack(X_channels, axis=-1) # Combine target frame L channel with colourised frame a+b channels

      new_frame = cnn.predict_lab(model, X_image) # Put through CNN to smooth colour differences
      hb.display_frame_pair(cv2.cvtColor(X_image, cv2.COLOR_Lab2BGR), cv2.cvtColor(new_frame, cv2.COLOR_Lab2BGR))
      new_frame_rgb = cv2.cvtColor(new_frame, cv2.COLOR_Lab2BGR)

      vid_out.write(new_frame_rgb) # Save frame to video
      prev_frame = new_frame

  vid_in.release()
  vid_out.release()

def main():
  t = time.time()
  # gts.training_set_from_video('/src/data/train_vids/gbh360.mp4', 5)
  # cnn.create()
  # gts.training_set_from_video('/src/test_vids/vid.mp4', 5, use_csv = True)
  # hb.split_video('/src/data/train_vids/grand_budapest_hotel.mp4', show_cuts = True, save_to_csv = True)
  colorise_video('/src/test_vids/vid.mp4', 10)

  # gts.save_images(train_X, '/src/data/train/', 'gbh-')
  # gts.save_images(train_y, '/src/data/train/', 'gbh-')
  elapsed = time.time() - t
  print(elapsed)

if __name__ == '__main__':
  main()