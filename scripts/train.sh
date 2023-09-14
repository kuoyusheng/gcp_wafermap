#!/bin/bash
gcloud ai custom-jobs local-run \
--executor-image-uri=us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13.py310:latest \
--local-package-path=. \
--python-module=trainer.task \
-- \
--train-files=gs://wmap-811k/WM811K.pkl \
--eval-files=gs://wmap-811k/WM811K.pkl