# docker build -t test . && docker run -it -p 8888:8888 -v .:/src test
# https://localhost:8888

FROM ashokponkumar/caffe-py3

WORKDIR /src
COPY ./requirements.txt /src/requirements.txt
RUN pip install -r requirements.txt

RUN mkdir ~/resources
RUN wget http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel -O ~/resources/colorization_release_v2.caffemodel
# RUN mv ./colorization_release_v2.caffemodel /src/zhang/models

CMD ["sh", "./setup.sh"]