#!/bin/bash
BASE_GPU_IMAGE='us-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.2-0:latest'
ARTIFACT_URI='gs://wmap-811k/model_20230831_041343'
gcloud ai models upload \
--region=us-west1 \
--display-name=test2 \
--container-image-uri=$BASE_GPU_IMAGE \
--artifact-uri=$ARTIFACT_URI