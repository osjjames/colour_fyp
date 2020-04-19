# docker build -t test . && docker run -it -p 8888:8888 -v .:/src test
# https://localhost:8888

# Dockerfile-gpu
FROM nvidia/cuda:10.0-cudnn7-runtime

# Installs necessary dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
         wget \
         curl \
         python3 \
         python3-distutils \
         libsm6 libxext6 libxrender-dev libglib2.0-0\
         build-essential && \
     rm -rf /var/lib/apt/lists/*

# Installs pip.
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py && \
    pip install setuptools && \
    rm get-pip.py

WORKDIR /src
COPY ./requirements.txt /src/requirements.txt
RUN pip install -r requirements.txt

# Installs cloudml-hypertune for hyperparameter tuning.
# It’s not needed if you don’t want to do hyperparameter tuning.
RUN pip install cloudml-hypertune

# Installs google cloud sdk, this is mostly for using gsutil to export model.
RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup

# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

# Copies the trainer code 
# RUN mkdir /src
COPY ./src/setup.sh /src/setup.sh
COPY ./src/cnn.py /src/cnn.py
COPY ./src/cnn_data_gen.py /src/cnn_data_gen.py
COPY ./src/config.py /src/config.py
COPY ./src/main.py /src/main.py
RUN mkdir /src/data
COPY ./src/data/train_names.txt /src/data/train_names.txt
COPY ./src/data/valid_names.txt /src/data/valid_names.txt
COPY ./src/data/pts_in_hull.npy /src/data/pts_in_hull.npy
RUN mkdir /src/data/train_X /src/data/train_y /src/data/models

CMD ["sh", "./setup.sh"]