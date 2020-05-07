#!/bin/bash
mv ~/resources/colorization_release_v2.caffemodel /resources
jupyter notebook --ip 0.0.0.0 --allow-root --no-browser --NotebookApp.token='test' --NotebookApp.max_buffer_size=4294967296