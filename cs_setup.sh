#!/bin/bash
cd /src/liblbfgs/
apt-get update
apt-get install -y libtool automake
./autogen.sh
./configure
make
make install


cd /src/pylbfgs/
python3 setup.py install