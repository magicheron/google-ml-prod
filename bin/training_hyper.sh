#!/bin/sh

gcloud ai-platform jobs submit training mnist_`date +"%s"` \
    --python-version 3.7 \
    --runtime-version 2.3 \
    --scale-tier BASIC \
    --package-path ./trainer \
    --module-name trainer.task \
    --region europe-west1 \
    --job-dir gs://mymagic-kschool-23012021/tmp \
    --config ./bin/hyper.yaml
    -- \
    --model-output-path gs://mymagic-kschool-23012021/models
    --hypertune