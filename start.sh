#!/bin/bash

docker run -it -p 8889:8888 -v "$(pwd):/home/jovyan/work" jupyter_container start-notebook.py --NotebookApp.token='my-token'

