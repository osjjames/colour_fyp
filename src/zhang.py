import sys
import matplotlib.pyplot as plt

sys.path.insert(1, '/src/zhang/colorization')

from colorize import colorize, colorize_from_file, colorize_from_grayscale

# img = colorize_from_file('/src/zhang/demo/imgs/dog.jpg')
# plt.imshow(img)