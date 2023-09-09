#!/bin/bash
gcloud ai endpoints deploy-model 1793884007042121728 \
  --region=us-west1 \
  --model=3994772034314829824 \
  --display-name=test2 \
  --min-replica-count=1 \
  --max-replica-count=2 \
  --traffic-split=0=100