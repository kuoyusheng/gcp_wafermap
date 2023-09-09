#!/bin/bash
torch-model-archiver --model-name model \
                     --version 0.1\
                     --model-file model.py \
                     --serialized-file model.pth \
                     --handler ssl_handler.py