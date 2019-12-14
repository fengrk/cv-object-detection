# -*- coding:utf-8 -*-

"""
    ref: https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#4-generate-training-data

"""

import logging
import os
import sys
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

from object_detection.model_main import main as train_main
from object_detection.utils import label_map_util, visualization_utils, ops as utils_ops

current_dir = os.path.abspath(os.path.dirname(__file__))

detection_data = namedtuple("detection_data", field_names=["filename", "xmin", "ymin", "xmax", "ymax", "prob"])


def get_image(path: str, width: int = None, height: int = None) -> Image:
    img = Image.open(path)
    if width is None and height is None:
        return img

    return img.resize((width, height), Image.ANTIALIAS)


def iter_list_with_size(src_list: list, size: int):
    """
        src_list would be modified when running
    """
    n_part = len(src_list) // size + 1
    while n_part >= 0:
        n_part -= 1
        part_src_list = src_list[:size]
        if part_src_list:
            yield part_src_list
            del src_list[:size]
        else:
            break


def parse_result(image: Image, file_name: str, output_dict: dict, min_score_thresh: float = 0.5) -> list:
    width, height = image.size
    result_list = []
    for i, boxes in enumerate(output_dict['detection_boxes']):
        if output_dict['detection_scores'][i] >= min_score_thresh:
            ymin, xmin, ymax, xmax = boxes
            result_list.append(
                detection_data(
                    file_name,
                    int(xmin * width), int(ymin * height),
                    int(xmax * width), int(ymax * height),
                    float(output_dict['detection_scores'][i])
                ))
        else:
            break

    return result_list


def load_image_into_numpy_array(_image):
    (im_width, im_height) = _image.size
    return np.array(_image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def train(model_config: str, train_ckpt_dir: str, train_steps: int = 30000, eval_steps: int = 500, ):
    raw_args = list(sys.argv)
    try:
        sys.argv = [
            "model_main.py",
            "--pipeline_config_path={}".format(model_config),
            "--model_dir={}".format(train_ckpt_dir),
            "--alsologtostderr",
            "--num_train_steps={}".format(train_steps),
            "--num_eval_steps={}".format(eval_steps),
        ]
        tf.app.run(main=train_main)
    finally:
        sys.argv = raw_args


def run_inference_image_list(image_list: list, graph: tf.Graph) -> list:
    """

    Args:
        image_list: list of np.ndarray. image in list must have the same shape
        graph:

    Returns:
        object:
    """
    shape = image_list[0].shape
    for image in image_list:
        assert shape == image.shape

    result_list = []
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image_list[0].shape[0], image_list[0].shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            for image in image_list:
                output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]

                result_list.append(output_dict)

    return result_list


def show_image_with_bounding_box(image_info: detection_data):
    image = Image.open(image_info.filename)
    visualization_utils.draw_bounding_box_on_image(
        image,
        image_info.ymin, image_info.xmin,
        image_info.ymax, image_info.xmax,
        use_normalized_coordinates=False,
    )
    image_np = load_image_into_numpy_array(image)
    plt.imshow(image_np)


def do_predict(run_inference, label_map: str, image_file_list: list, show_image: bool = False, batch_size: int = 1) -> list:
    # What model to download.
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    num_classes = 1

    label_map = label_map_util.load_labelmap(label_map)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Size, in inches, of the output images.
    image_size = (8, 8)

    result_list = []

    image_count = len(image_file_list)
    _step = 0
    for image_path_list in iter_list_with_size(image_file_list, size=batch_size):
        _image_obj_list = [get_image(image_path, width=344, height=344) for image_path in image_path_list]
        image_np_list = [load_image_into_numpy_array(image) for image in _image_obj_list]

        output_dict_list = run_inference(image_np_list)

        _step += len(image_path_list)
        logging.info("predict {}/{}".format(_step, image_count))

        for i, _image_file in enumerate(image_path_list):
            _image = Image.open(_image_file)
            image_np = load_image_into_numpy_array(_image)
            output_dict = output_dict_list[i]

            result_list.extend(
                parse_result(_image, file_name=_image_file, output_dict=output_dict)
            )

            if show_image:
                # Visualization of the results of a detection.
                visualization_utils.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    output_dict['detection_boxes'],
                    output_dict['detection_classes'],
                    output_dict['detection_scores'],
                    category_index,
                    instance_masks=output_dict.get('detection_masks'),
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=1000,
                    line_thickness=4)

                plt.figure(figsize=image_size)
                plt.imshow(image_np)

    return result_list


def predict(frozen_inference_dir: str, label_map: str, image_file_list: list, show_image: bool = False, batch_size: int = 1) -> list:
    # What model to download.
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    inference_graph = os.path.join(frozen_inference_dir, 'frozen_inference_graph.pb')

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(inference_graph, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    def run_inference(_image_np_list):
        return run_inference_image_list(image_list=_image_np_list, graph=detection_data)

    return do_predict(run_inference, label_map=label_map, image_file_list=image_file_list, show_image=show_image, batch_size=batch_size)


def save_detection_result(result_list: list, output_file: str):
    with open(output_file, "w", encoding="utf-8") as f:
        for image_info in result_list:
            f.write("{},{} {} {} {}\n".format(
                os.path.basename(image_info.filename),
                image_info.xmin, image_info.ymin,
                image_info.xmax, image_info.ymax,
            ))
