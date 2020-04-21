#!/bin/bash
# gsutil -md cp -r gs://osjjames-aiplatform/train_X/train_X/gbh360-0.png /src/data/train_X/gbh360-0.png
# ls /src/data/train_X/
gsutil -mq cp -r gs://osjjames-aiplatform/train_X/train_X/ /src/data/
gsutil -mq cp -r gs://osjjames-aiplatform/train_y/train_y/ /src/data/
# ls /src/data/train_X/
# ls /src/data/train_y/

echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64/" >> /root/.bashrc
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64" >> /root/.bashrc

python3 main.py