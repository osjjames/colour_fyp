#!/bin/bash
gsutil -md cp -r gs://osjjames-aiplatform/train_X/train_X/gbh360-0.png /src/data/train_X/gbh360-0.png
ls /src/data/train_X/ 
# gsutil -m cp -r gs://osjjames-aiplatform/train_X/train_X/ /src/data/train_X/
# gsutil -m cp -r gs://osjjames-aiplatform/train_y/train_y/ /src/data/train_y/
ldconfig -p | grep cublas
python3 main.py