# coding:utf-8
__author__ = 'rk.feng'

import matplotlib
import numpy as np
import tensorflow as tf
from PIL import Image
from grpc.beta import implementations
from tensorboard._vendor.tensorflow_serving.apis import prediction_service_pb2, predict_pb2

from tf_utils import load_image_into_numpy_array, do_predict

matplotlib.use('tkagg')


def call_tf_serving(image_list: [str], server: str):
    """ """
    # Create stub
    image_np = np.zeros(shape=(len(image_list), 344, 344, 3), dtype=np.uint8)
    if isinstance(image_list[0], str):
        for i in range(len(image_list)):
            image_np[i] = load_image_into_numpy_array(Image.open(image_list[i]))
    else:
        for i in range(len(image_list)):
            image_np[i] = image_list[i]

    host, port = server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    # Create prediction request object
    request = predict_pb2.PredictRequest()

    # Specify model name (must be the same as when the TensorFlow serving serving was started)
    request.model_spec.name = 'object'

    # Initalize prediction
    # Specify signature name (should be the same as specified when exporting model)
    request.model_spec.signature_name = "serving_default"
    request.inputs['inputs'].CopyFrom(
        tf.contrib.util.make_tensor_proto(image_np, shape=image_np.shape))

    # Call the prediction server
    return stub.Predict(request, 20.0 * len(image_list))  # 20 secs timeout


def tf_serving_predict(image_list: [str], server: str, label_map: str, show_image: bool = False, batch_size: int = 2):
    """ """

    def run_inference(_image_np_list):
        _result = call_tf_serving(image_list=_image_np_list, server=server)
        _dict = {
            "detection_boxes": np.reshape(_result.outputs['detection_boxes'].float_val, newshape=(-1, 4)),
            "detection_classes": np.squeeze(_result.outputs['detection_classes'].float_val).astype(np.int32),
            "detection_scores": np.squeeze(_result.outputs['detection_scores'].float_val),
        }
        length = len(_image_np_list)
        for key, value in _dict.items():
            _new_shape = list(value.shape)
            _new_shape.insert(0, -1)
            _new_shape[1] = _new_shape[1] // length
            _dict[key] = np.reshape(value, newshape=_new_shape)

        output_dict_list = []
        for i in range(length):
            output_dict_list.append({
                "detection_boxes": _dict['detection_boxes'][i],
                "detection_classes": _dict['detection_classes'][i],
                "detection_scores": _dict['detection_scores'][i],
            })

        return output_dict_list

    return do_predict(run_inference=run_inference, label_map=label_map, image_file_list=image_list, show_image=show_image, batch_size=batch_size)
