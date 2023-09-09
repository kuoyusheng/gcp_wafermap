#!/bin/bash
BASE_GPU_IMAGE='us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13.py310:latest'
gcloud ai custom-jobs create \
--region=us-west1 \
--display-name=test2 \
--worker-pool-spec=machine-type=n1-standard-4,replica-count=1,executor-image-uri=$BASE_GPU_IMAGE,local-package-path=.,python-module=trainer.task \
--args=--train-files=/gcs/wmap-811k/WM811K.pkl \
--args=--eval-files=/gcs/wmap-811k/WM811K.pkl \
--args=--job-dir=gs://wmap-811k \
--args=--num-epochs=1
#