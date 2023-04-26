from tensorflow import keras as tfk


def mobilenet_v3(
    input_shape=(224, 224, 3), num_classes=2, version=None, multilabel=False, **kwargs
):

    model_name = "mobilenet_v3_classifier"
    if version:
        model_name = version

    backbone = tfk.applications.MobileNetV3Large(
        input_shape=input_shape, include_top=False, pooling="avg"
    )
    backbone.trainable = False
    inputs = tfk.Input(shape=input_shape, name="inputs")
    g_inputs = tfk.layers.GaussianNoise(stddev=0.01)(inputs)
    x = backbone(g_inputs)
    if multilabel:
        o = tfk.layers.Dense(
            num_classes, activation="sigmoid", name="classification_head"
        )(x)
    else:
        o = tfk.layers.Dense(
            num_classes, activation="softmax", name="classification_head"
        )(x)
    model = tfk.Model(inputs=inputs, outputs=[o], name=model_name)
    return model


def mobilenet_v3_small(
    input_shape=(224, 224, 3), num_classes=2, version=None, multilabel=False, **kwargs
):

    model_name = "mobilenet_v3_small_classifier"
    if version:
        model_name = version

    backbone = tfk.applications.MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        pooling="avg",
        alpha=1.0,
        minimalistic=False,
    )
    backbone.trainable = False
    inputs = tfk.layers.Input(shape=input_shape)
    x = backbone(inputs, training=False)

    if multilabel:
        o = tfk.layers.Dense(
            num_classes, activation="sigmoid", name="classification_head"
        )(x)
    else:
        o = tfk.layers.Dense(
            num_classes, activation="softmax", name="classification_head"
        )(x)
    model = tfk.Model(inputs=inputs, outputs=[o], name=model_name)
    return model


def build_mobilenet(input_shape, num_classes):

    model_name = "clf_mobilenet_v3"

    input_tensor = tfk.layers.InputLayer(input_shape=input_shape, name="inputs")
    backbone = tfk.applications.MobileNetV3Large(
        input_shape=input_shape, include_top=False, pooling="max"
    )
    backbone.trainable = True
    num_layers = len(backbone.layers)

    for ix in range(num_layers):
        if ix == num_layers - 6:
            backbone.layers[ix].trainable = True
            # backbone.layers[ix]._name = "last_conv_layer"
        else:
            backbone.layers[ix].trainable = False
    inputs = input_tensor.input

    x = backbone(inputs)

    o = tfk.layers.Dense(num_classes, activation="softmax", name="classification_head")(
        x
    )
    model = tfk.Model(inputs=inputs, outputs=[o], name=model_name)
    return model


def build_mobilenet_w_clf_head(input_shape, act, dropout_perc, num_classes):

    model_name = "clf_mobilenet_v3"
    input_tensor = tfk.layers.InputLayer(input_shape=input_shape, name="inputs")
    backbone = tfk.applications.MobileNetV3Large(
        input_shape=input_shape, include_top=False, pooling="max"
    )
    backbone.trainable = True

    num_layers = len(backbone.layers)
    for ix in range(num_layers):
        if (ix == num_layers - 6) or (ix == num_layers - 13):
            backbone.layers[ix].trainable = True
            # backbone.layers[ix]._name = "last_conv_layer"
        else:
            backbone.layers[ix].trainable = False
    inputs = input_tensor.input

    x = backbone(inputs)
    x = tfk.layers.BatchNormalization(momentum=0.9)(x)

    x = tfk.layers.Dense(512, activation=act)(x)
    x = tfk.layers.Dropout(dropout_perc)(x)
    x = tfk.layers.Dense(128, activation=act)(x)
    x = tfk.layers.Dropout(dropout_perc)(x)
    x = tfk.layers.Dense(32, activation=act)(x)
    x = tfk.layers.Dropout(dropout_perc)(x)

    o = tfk.layers.Dense(num_classes, activation="softmax", name="classification_head")(
        x
    )
    model = tfk.Model(inputs=inputs, outputs=[o], name=model_name)
    return model


def build_mobilenet_small(input_shape, num_classes):

    model_name = "clf_mobilenet_v3_small"
    input_tensor = tfk.layers.InputLayer(input_shape=input_shape, name="inputs")
    backbone = tfk.applications.MobileNetV3Small(
        input_shape=input_shape, include_top=False, pooling="max"
    )
    backbone.trainable = True

    num_layers = len(backbone.layers)
    for ix in range(num_layers):
        if ix == num_layers - 6:
            backbone.layers[ix].trainable = True
        else:
            backbone.layers[ix].trainable = False
    inputs = input_tensor.input

    x = backbone(inputs)

    o = tfk.layers.Dense(num_classes, activation="softmax", name="classification_head")(
        x
    )
    model = tfk.Model(inputs=inputs, outputs=[o], name=model_name)
    return model


if __name__ == "__main__":
    # backbone = tfk.applications.MobileNetV3Small(
    #     input_shape=(224, 224, 3), include_top=False, pooling="max"
    # )

    # backbone.summary()

    # net = mobilenet_v3_small(trainable_conv=1)
    # net.compile(optimizer="adam", loss="binary_crossentropy")
    # net.summary()

    import tensorflow as tf

    backbone = tf.tfk.applications.EfficientNetB4(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=(224, 224, 3),
        pooling="max",
    )
    backbone.summary()
