# coding:utf-8

__author__ = 'rk.feng'

import glob
import hashlib
import io
import logging
import os
import random
from collections import namedtuple

import pandas as pd
import tensorflow as tf
from PIL import Image

from object_detection.utils import dataset_util

current_dir = os.path.abspath(os.path.dirname(__file__))

_cur_dir = os.path.dirname(__file__)
DATA_DIR = os.path.abspath(os.path.join(_cur_dir, "../dataset"))

# setting
train_validate_image_dir = os.path.join(DATA_DIR, "train")
raw_train_csv = os.path.join(DATA_DIR, "train_labels.csv")
train_csv = os.path.join(DATA_DIR, "train.csv")
validate_csv = os.path.join(DATA_DIR, "validate.csv")
train_tf_file = os.path.join(DATA_DIR, "train.tf.record")
validate_tf_file = os.path.join(DATA_DIR, "validate.tf.record")


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(1 if row['class'] == "word" else None)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def create_tf_record_file():
    def split(_df, _group):
        data = namedtuple('data', ['filename', 'object'])
        _gb = _df.groupby(_group)
        return [data(filename, _gb.get_group(x)) for filename, x in zip(_gb.groups.keys(), _gb.groups)]

    if os.path.exists(train_tf_file) and os.path.exists(validate_tf_file):
        print("{} and {} exists before! No need to create it again!".format(train_tf_file, validate_tf_file))
        return

    # train
    with tf.python_io.TFRecordWriter(train_tf_file) as writer:
        grouped = split(pd.read_csv(train_csv), 'filename')
        for group in grouped:
            tf_example = create_tf_example(group, train_validate_image_dir)
            writer.write(tf_example.SerializeToString())

        logging.info('Successfully created the TFRecords: {}'.format(train_tf_file))

    # test
    with tf.python_io.TFRecordWriter(validate_tf_file) as writer:
        grouped = split(pd.read_csv(validate_csv), 'filename')
        for group in grouped:
            tf_example = create_tf_example(group, train_validate_image_dir)
            writer.write(tf_example.SerializeToString())

        logging.info('Successfully created the TFRecords: {}'.format(validate_tf_file))


def crop_images():
    """ 裁边 """
    image_dir_list = [os.path.abspath(os.path.join(DATA_DIR, _dir)) for _dir in ["train", "test"]]

    for image_dir in image_dir_list:
        image_file_list = glob.glob("{}/*.jpg".format(image_dir))
        print("there are {} image in dir {}".format(len(image_file_list), image_dir))

        for image_file in image_file_list:
            original = Image.open(image_file)

            width, height = original.size  # Get dimensions
            assert (height - 40) == width
            cropped_example = original.crop((0, 0, width, height - 40))
            cropped_example.save(image_file)

        print("{} image cropped and saved in dir: {}".format(len(image_file_list), image_dir))


def create_train_validate():
    """ 分割数据库 """
    if os.path.exists(train_csv) and os.path.exists(validate_csv):
        print("train.csv and validate.csv exists before! No need to create it again!")
        return

    train_info = pd.read_csv(raw_train_csv)

    # split train/test from train dataset
    image_list = glob.glob("{}/*.jpg".format(os.path.join(DATA_DIR, "train")))
    assert len(image_list) > 0

    # split by type
    type_cache = {}
    for image_file in image_list:
        _image_type = "-".join(os.path.basename(image_file).split("-")[:-1])
        type_cache.setdefault(_image_type, []).append(image_file)

    train_image_list = []
    for _image_type, _image_list in type_cache.items():
        _image_list.sort()
        random.seed(int(hashlib.md5(_image_type.encode("utf-8")).hexdigest(), 16))
        train_image_list.extend(random.sample(_image_list, int(len(_image_list) * 0.8)))

    train_image_file_set = [os.path.basename(file_name) for file_name in train_image_list]

    # calc csv content
    _head_line = "filename,width,height,class,xmin,ymin,xmax,ymax"
    train_csv_content_list, test_csv_content_list = [_head_line, ], [_head_line, ]
    for i, file_name in enumerate(train_info.ID):
        record = "{},344,344,word,{}".format(
            file_name, ",".join(train_info[" Detection"][i].split(" "))
        )
        if file_name in train_image_file_set:
            train_csv_content_list.append(record)
        else:
            test_csv_content_list.append(record)

    # save csv file
    with open(train_csv, "w", encoding='utf-8') as f:
        f.write("\n".join(train_csv_content_list) + "\n")

    with open(validate_csv, "w", encoding='utf-8') as f:
        f.write("\n".join(test_csv_content_list) + "\n")


if __name__ == '__main__':
    # 图片裁剪
    crop_images()

    # 划分数据集
    create_train_validate()

    # build tf record file
    create_tf_record_file()
