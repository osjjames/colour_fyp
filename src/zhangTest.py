import sys
import matplotlib.pyplot as plt

sys.path.insert(1, '/src/zhang/colorization')

import colorize

img = colorize.colorize('/src/zhang/demo/imgs/dog.jpg')
plt.imshow(img)