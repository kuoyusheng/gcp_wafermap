#!/bin/bash
gcloud ai custom-jobs create \
--region=us-west4 \
--display-name=test \
--config=config.yaml
#--worker-pool-spec=machine-type=n1-standard-4,replica-count=1,executor-image-uri=us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13.py310:latest,local-package-path=.,python-module=trainer.task \
#--args=--train-files=/gcs/wmap-811k/WM811K.pkl \
#--args=--eval-files=/gcs/wmap-811k/WM811K.pkl
