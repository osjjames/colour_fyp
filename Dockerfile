# docker build -t test . && docker run -it -p 8888:8888 -v .:/src test
# https://localhost:8888

FROM ashokponkumar/caffe-py3

WORKDIR /src
COPY ./requirements.txt /src/requirements.txt
RUN pip install -r requirements.txt

COPY ./src/liblbfgs /src/liblbfgs
COPY ./src/pylbfgs /src/pylbfgs
COPY ./cs_setup.sh /src/cs_setup.sh
RUN sh ./cs_setup.sh
# ADD . /src
# EXPOSE 8888
CMD ["sh", "./setup.sh"]
# CMD ["jupyter", "notebook", "--ip", "0.0.0.0", "--allow-root", "--no-browser", "--NotebookApp.token='test'"]
# CMD [ "python", "./main.py"]