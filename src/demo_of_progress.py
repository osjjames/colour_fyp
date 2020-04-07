import histogram_test as ht
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import zhang

def split_colourise():
  groups = ht.split_video('/src/test_vids/vid.mp4')
  sample_frame_indices = np.random.choice(len(groups[0]), int(np.floor(0.1 * len(groups[0]))), replace=False)
  for i in tqdm_notebook(sample_frame_indices):
    col_frame = zhang.colorize_from_grayscale(groups[0][i])

    plt.imshow(col_frame)
    plt.show()

def split():
  ht.split_video('/src/test_vids/vid.mp4')

def cs():
  import compressed_sensing
