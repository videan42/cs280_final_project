#!/bin/bash

set -e
set -o nounset

cd py-faster-rcnn/caffe-fast-rcnn
rm -rf ./build
mkdir ./build
cd ./build

cmake -DBLAS=Open \
    -DHDF5_DIR=/usr/lib \
    -DCUDA_TOOLKIT_ROOT_DIR=/opt/cuda \
    ..

make -j8
make pycaffe
