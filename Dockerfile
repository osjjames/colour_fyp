# docker build -t test . && docker run -it -p 8888:8888 -v .:/src test
# https://localhost:8888

FROM tensorflow/tensorflow:latest-py3-jupyter

WORKDIR /src
ADD ./requirements.txt /src/requirements.txt
RUN pip install -r requirements.txt
# ADD . /src
CMD ["jupyter", "notebook", "--ip", "0.0.0.0", "--allow-root", "--no-browser"]
# CMD [ "python", "./main.py"]