import tensorflow as tf
from tensorflow import keras as tfk


def resnet50_v2(
    input_shape=(224, 224, 3),
    num_classes=2,
    version=None,
    trainable_backbone=False,
    **kwargs
):

    model_name = "resnet50_v2_classifier"
    if version:
        model_name = version

    input_tensor = tfk.layers.Input(input_shape, dtype=tf.float32)

    x = tfk.applications.resnet_v2.preprocess_input(input_tensor)
    backbone = tfk.applications.ResNet50V2(
        include_top=False, weights="imagenet", input_tensor=x, pooling="avg"
    )
    backbone.trainable = False  # trainable_backbone

    inputs = tfk.layers.Input(shape=input_shape)
    x = backbone(inputs, training=False)
    o = tfk.layers.Dense(num_classes, activation="softmax", name="classification_head")(
        x
    )
    model = tfk.Model(inputs=inputs, outputs=[o], name=model_name)
    return model


if __name__ == "__main__":
    model = resnet50_v2()
    model.summary()
