import time
from tqdm.auto import tqdm, trange
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

import cnn
import zhang
import gen_training_set as gts
import shot_cut as sc
import video_utils as vu

def colorise_video(video_path, n):
  save_path = os.path.splitext(video_path)[0] + '-colourised-' + str(n) + '.mp4'
  vid_out = vu.setup_writer(video_path, save_path) # Setup video writer to save output to file

  model = cnn.load_model(cnn.checkpoint_models_path + "full_model_256.hdf5") # Load CTCNN model

  shot_indices = sc.split_video(video_path) # Split input video by shots

  vid_in = cv2.VideoCapture(video_path)  # Import video

  for shot_number in trange(len(shot_indices), desc="Group Number"):
    curr_shot_Lab = gts.read_group_frames_lab(shot_indices, shot_number, vid_in) # Read all frames of current shot in CIELAB format

    sample_frame_indices = []
    radius = int(np.floor(n/2)) # every sample frame should have a radius of n/2 intermediate frames in both directions
    for i in range(radius,len(curr_shot_Lab),n): # choose every nth frame to be a sample frame
      sample_frame_indices.append(i)

    for j in tqdm(sample_frame_indices, desc="Color Frame Batch"):
      sample_frame = zhang.colorize(curr_shot_Lab[j], lab_only = True) # Colourise every nth frame using Zhang et al.'s CNN

      colour_a = sample_frame[:,:,1] # Extract a + b channels
      colour_b = sample_frame[:,:,2]
      for k in range(j-radius, j+radius+1): # For all frames in current bundle
        if k != j and k < len(curr_shot_Lab): # If the frame is an intermediate frame
          intermediate = curr_shot_Lab[k]
          intermediate_l = intermediate[:,:,0] # Extract L channel

          X_channels = [intermediate_l, colour_a, colour_b]
          X_image = np.stack(X_channels, axis=-1) # Combine intermediate frame L channel with sample frame a+b channels

          new_frame = cnn.predict_lab(model, X_image) # Put through CTCNN to smooth colour differences
          new_frame_rgb = cv2.cvtColor(new_frame, cv2.COLOR_Lab2BGR) # Convert back to BGR to write to video

        if k == j and k < len(curr_shot_Lab): # If the frame is a sample frame
          new_frame_rgb = cv2.cvtColor(sample_frame, cv2.COLOR_Lab2BGR)

        vid_out.write(new_frame_rgb) # Save frame to video

  vid_in.release()
  vid_out.release() # Save final output video

def colorise_video_one(video_path):
  save_path = os.path.splitext(video_path)[0] + '-colourised-one.mp4'
  vid_out = vu.setup_writer(video_path, save_path)

  model = cnn.load_model(cnn.checkpoint_models_path + "full_model_256.hdf5") # Load CNN model
  print('model loaded')

  group_indices = sc.split_video(video_path) # Split input video by shots

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
      # sc.display_frame_pair(cv2.cvtColor(X_image, cv2.COLOR_Lab2BGR), cv2.cvtColor(new_frame, cv2.COLOR_Lab2BGR))
      new_frame_rgb = cv2.cvtColor(new_frame, cv2.COLOR_Lab2BGR)

      vid_out.write(new_frame_rgb) # Save frame to video
      prev_frame = new_frame

  vid_in.release()
  vid_out.release()

def colourise_and_record(video_path, start_n, end_n):
  path_prefix = os.path.splitext(video_path)[0]
  data_save_path = path_prefix + '-data.txt'
  data_file = open(data_save_path, 'a')

  t = time.time()
  frame_count = sc.split_video(video_path, use_csv = False, return_frame_count = True)
  elapsed = time.time() - t
  data_file.write("Frame count:\n")
  data_file.write(str(frame_count) + "\n")
  data_file.write("Split time:\n")
  data_file.write(str(elapsed) + "\n")

  for n in trange(start_n, end_n+1, 2):
    t = time.time()
    colorise_video(video_path, n)
    elapsed = time.time() - t
    data_file.write("Colourisation time (n = %d):\n" % n)
    data_file.write(str(elapsed) + "\n")

  data_file.close()

def main():
  t = time.time()

  # names = ['car', 'ewan', 'desert', 'gandhi', 'cafe', 'italy', 'rd_shoot', 'mail', 'phone', 'park']
  # for name in tqdm(names):
  # name = 'car'
  # colourise_and_record('/src/data/train_vids/' + name + '.avi', 3, 9)

  # gts.training_set_from_video('/src/data/train_vids/gbh360.mp4', 5)
  # cnn.create()
  # gts.training_set_from_video('/src/test_vids/vid.mp4', 5, use_csv = True)
  # sc.split_video('/src/data/train_vids/grand_budapest_hotel.mp4', show_cuts = True, save_to_csv = True)
  colorise_video('/src/data/train_vids/grand_budapest_hotel.mp4', 9)

  # gts.save_images(train_X, '/src/data/train/', 'gbh-')
  # gts.save_images(train_y, '/src/data/train/', 'gbh-')

  elapsed = time.time() - t
  print(elapsed)

if __name__ == '__main__':
  main()