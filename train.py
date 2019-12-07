# coding:utf-8
__author__ = 'rk.feng'

import os
import re

from src.dataset_utils import validate_csv, validate_tf_file, train_tf_file
from tf_utils import train
from x_logger import global_init_logger

_cur_dir = os.path.dirname(__file__)
global_init_logger(log_file="tf.log")


def create_config_file(config_file: str, pretrained_model_dir: str):
    file_name = os.path.join(_cur_dir, "object_detection/samples/configs/faster_rcnn_inception_v2_pets.config")

    with open(file_name, "r", encoding="utf-8") as f:
        s = f.read()

    with open(config_file, 'w', encoding="utf-8") as f:
        s = re.sub('num_classes: 37', "num_classes: 1", s)
        s = re.sub('min_dimension: 600', "min_dimension: 300", s)
        s = re.sub('max_dimension: 1024', "max_dimension: 380", s)
        s = re.sub('first_stage_max_proposals: 300', "first_stage_max_proposals: 300", s)
        s = re.sub('max_detections_per_class: 100', "max_detections_per_class: 10", s)
        s = re.sub('max_total_detections: 300', "max_total_detections: 10", s)

        s = re.sub('PATH_TO_BE_CONFIGURED/model.ckpt', '{}/model.ckpt'.format(os.path.abspath(pretrained_model_dir)), s)

        s = re.sub('PATH_TO_BE_CONFIGURED/pet_faces_train.record-\?\?\?\?\?-of-00010', train_tf_file, s)

        s = re.sub('PATH_TO_BE_CONFIGURED/pet_faces_val.record-\?\?\?\?\?-of-00010', validate_tf_file, s)

        # 验证集图片数
        validate_image_files = set()
        with open(validate_csv, "r") as _f:
            for line in _f:
                if line.strip():
                    validate_image_files.add(line.strip().split(",")[0])
        s = re.sub('num_examples: 1101', "num_examples: {}".format(len(validate_image_files)), s)

        s = re.sub('PATH_TO_BE_CONFIGURED/pet_label_map.pbtxt', os.path.abspath(os.path.join(_cur_dir, "labelmap.pbtxt")), s)

        f.write(s)


if __name__ == '__main__':
    model_dir = os.path.join(_cur_dir, "models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    model_config = os.path.join(_cur_dir, "faster_rcnn_inception_v2_pets.config")
    if not os.path.exists(model_config):
        create_config_file(config_file=model_config, pretrained_model_dir=os.path.join(_cur_dir, "pretrained_models"))

    train(
        model_config=model_config,
        train_ckpt_dir=model_dir,
    )
