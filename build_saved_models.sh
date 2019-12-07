#!/usr/bin/env zsh

last_model="./models/model.ckpt-24841"
fine_tuned_model="./data/output"
config_file="./faster_rcnn_inception_v2_pets.config"

PYTHONPATH=$(pwd) python object_detection/export_inference_graph.py \
    --input_type=image_tensor \
    --pipeline_config_path=$config_file \
    --output_directory=$fine_tuned_model \
    --trained_checkpoint_prefix=$last_model \
    --saved_model_with_variables=True