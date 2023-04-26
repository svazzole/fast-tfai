# import numpy as np

from typing import Callable

import tensorflow as tf

import fast_tfai.models.mobilenet as fast_mobi
from fast_tfai.models.densenet import build_densenet121
from fast_tfai.models.depthwise import build_depthwise, build_depthwise_small
from fast_tfai.models.resnet import resnet50_v2
from fast_tfai.utils.console import console
from fast_tfai.utils.validate_args import TrainerConf


class ModelBuilder:
    registry = {
        "depthwise_small": build_depthwise_small,
        "mobilenet": fast_mobi.build_mobilenet,
        "mobilenet_v3": fast_mobi.mobilenet_v3,
        "mobilenet_v3_small": fast_mobi.mobilenet_v3_small,
        "depthwise": build_depthwise,
        "densenet121": build_densenet121,
        "resnet50_v2": resnet50_v2,
    }

    @classmethod
    def register(cls, name: str) -> Callable:
        def inner_wrapper(builder_func) -> Callable:
            if name in cls.registry:
                console.log(f"Model {name} already exists. Will replace it")
            cls.registry[name] = builder_func
            return builder_func

        return inner_wrapper

    @classmethod
    def build(cls, name: str, *args, **kwargs) -> tf.keras.Model:
        if name not in cls.registry.keys():
            raise ValueError("Model not available in Model Registry")
        return cls.registry[name](*args, **kwargs)

    @classmethod
    def from_conf(cls, conf: TrainerConf, num_classes: int = 2) -> tf.keras.Model:
        name = conf.model.name
        params = conf.model.params
        params["version"] = conf.version
        params["num_classes"] = num_classes
        multilabel = False
        if conf.task == "multilabel":
            multilabel = True
        params["multilabel"] = multilabel
        return cls.build(name=name, **params)

    @staticmethod
    def wrap_with_heatmap(model):
        return _wrap_with_heatmap(model)


def _wrap_with_heatmap(model: tf.keras.Model):
    last_conv_layer = model.layers[-2].layers[-2].get_output_at(0)
    weights = model.layers[-1].get_weights()[0]

    backbone_conv_model = tf.keras.Model(
        inputs=model.layers[-2].input, outputs=[last_conv_layer]
    )

    new_inputs = tf.keras.layers.Input(shape=model.input.shape[1:])
    conv_layer = backbone_conv_model(new_inputs)
    new_model = tf.keras.Model(inputs=new_inputs, outputs=[conv_layer], name="castrato")

    new_input = tf.keras.layers.Input(shape=model.input.shape[1:])
    pred = model(new_input)
    conv = new_model(new_input)
    pred_class = tf.math.argmax(pred, axis=1)

    reshaped_w = tf.reshape(weights, [-1, *weights.shape])

    class_weights = tf.transpose(
        tf.gather(reshaped_w, pred_class, axis=2), perm=[2, 1, 0]
    )

    output = tf.einsum("bijc,bck->bijk", conv, class_weights)

    heatmap = tf.cast(
        (output - tf.reduce_min(output, axis=[1, 2], keepdims=True))
        / (
            tf.reduce_max(output, axis=[1, 2], keepdims=True)
            - tf.reduce_min(output, axis=[1, 2], keepdims=True)
        )
        * 255,
        dtype=tf.uint8,
    )

    heatmap = tf.image.resize(
        heatmap, model.input.shape[1:3], method="bilinear", name="heatmap"
    )

    return tf.keras.Model(new_input, [pred, heatmap])


def _wrap_with_heatmap_multilabel(model: tf.keras.Model):
    last_conv_layer = model.layers[-2].layers[-2].get_output_at(0)
    weights = model.layers[-1].get_weights()[0]

    backbone_conv_model = tf.keras.Model(
        inputs=model.layers[-2].input, outputs=[last_conv_layer]
    )

    new_inputs = tf.keras.layers.Input(shape=model.input.shape[1:])
    conv_layer = backbone_conv_model(new_inputs)
    new_model = tf.keras.Model(inputs=new_inputs, outputs=[conv_layer], name="castrato")

    new_input = tf.keras.layers.Input(shape=model.input.shape[1:])
    pred = model(new_input)
    conv = new_model(new_input)

    mask = pred > 0.5
    pred_proba = pred * tf.cast(mask, tf.float32)
    normalized_proba = pred_proba / tf.reduce_sum(pred_proba, axis=1, keepdims=True)
    normalized_proba = tf.where(
        tf.math.is_nan(normalized_proba),
        tf.zeros_like(normalized_proba),
        normalized_proba,
    )

    weighted_weights = tf.matmul(normalized_proba, tf.transpose(weights, perm=[1, 0]))
    heatmap = tf.matmul(conv, tf.expand_dims(weighted_weights, axis=2))
    heatmap = (
        (heatmap - tf.reduce_min(heatmap))
        / (tf.reduce_max(heatmap) - tf.reduce_min(heatmap))
        * 255
    )

    heatmaps = tf.image.resize(
        heatmap, model.input.shape[1:3], method="bilinear", name="heatmap"
    )

    return tf.keras.Model(new_input, [pred, heatmaps])
