#!/bin/bash
docker run --gpus 1 -p 9090:9090 -it -v $(pwd):/DREAMPlace dreamplace2:cuda bash
