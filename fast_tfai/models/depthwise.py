import tensorflow as tf
import tensorflow.keras as tfk


def build_depthwise(
    input_shape, act, dropout_perc, num_classes, num_channels, momentum, epsilon
):

    model_name = "depthwise"

    inputs = tfk.Input(shape=input_shape, name="inputs")
    x = tf.image.per_image_standardization(inputs)
    x = tfk.layers.GaussianNoise(stddev=0.01)(x)

    x = tfk.layers.SeparableConv2D(
        num_channels,
        (3, 3),
        1,
        padding="same",
    )(x)
    x = act(x)
    x = tfk.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = tfk.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)(x)

    x = tfk.layers.SeparableConv2D(
        2 * num_channels,
        (3, 3),
        1,
        padding="same",
    )(x)
    x = act(x)
    x = tfk.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = tfk.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)(x)

    x = tfk.layers.SeparableConv2D(
        4 * num_channels,
        (3, 3),
        1,
        padding="same",
    )(x)
    x = act(x)
    x = tfk.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = tfk.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)(x)

    x = tfk.layers.SeparableConv2D(
        8 * num_channels,
        (3, 3),
        1,
        padding="same",
        name="last_conv",
    )(x)
    x = act(x)
    x = tfk.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)(x)

    x = tfk.layers.GlobalMaxPooling2D()(x)

    x = tfk.layers.Dense(2 * num_channels)(x)
    x = act(x)
    x = tfk.layers.Dropout(dropout_perc)(x)

    x = tfk.layers.Dense(num_channels)(x)
    x = act(x)
    x = tfk.layers.Dropout(dropout_perc)(x)

    x = tfk.layers.Dense(num_classes, activation="softmax", name="classification_head")(
        x
    )

    return tfk.models.Model(inputs=inputs, outputs=[x], name=model_name)


def build_depthwise_v2(
    input_shape, act, dropout_perc, num_classes, num_channels, momentum, epsilon
):

    model_name = "depthwise_v2"

    inputs = tfk.Input(shape=input_shape, name="inputs")
    x = tf.image.per_image_standardization(inputs)
    x = tfk.layers.GaussianNoise(stddev=0.01)(x)

    x = tfk.layers.SeparableConv2D(num_channels, (3, 3), 1, padding="same")(x)
    x = act(x)
    x = tfk.layers.SeparableConv2D(2 * num_channels, (3, 3), 2, padding="same")(x)
    x = tfk.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)(x)
    x = act(x)
    x = tfk.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = tfk.layers.SeparableConv2D(2 * num_channels, (3, 3), 1, padding="same")(x)
    x = act(x)
    x = tfk.layers.SeparableConv2D(4 * num_channels, (3, 3), 2, padding="same")(x)
    x = tfk.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)(x)
    x = act(x)
    x = tfk.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = tfk.layers.SeparableConv2D(4 * num_channels, (3, 3), 1, padding="same")(x)
    x = act(x)
    x = tfk.layers.SeparableConv2D(
        8 * num_channels, (3, 3), 2, padding="same", name="last_conv"
    )(x)
    x = tfk.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)(x)
    x = act(x)

    x = tfk.layers.GlobalMaxPooling2D()(x)

    x = tfk.layers.Dense(2 * num_channels)(x)
    x = act(x)
    x = tfk.layers.Dropout(dropout_perc)(x)

    x = tfk.layers.Dense(num_channels)(x)
    x = act(x)
    x = tfk.layers.Dropout(dropout_perc)(x)

    x = tfk.layers.Dense(num_classes, activation="softmax", name="classification_head")(
        x
    )

    return tfk.models.Model(inputs=inputs, outputs=[x], name=model_name)


def build_depthwise_small(
    input_shape=(224, 224, 3),
    activations="relu",
    num_classes=2,
    num_channels=32,
    momentum=0.99,
    epsilon=0.001,
    version=None,
    **kwargs
):

    model_name = "depthwise_small"
    if version:
        model_name = version
    act = tfk.layers.Activation(activations)

    inputs = tfk.Input(shape=input_shape, name="inputs")
    x = tf.image.per_image_standardization(inputs)
    x = tfk.layers.GaussianNoise(stddev=0.01)(x)

    x = tfk.layers.SeparableConv2D(num_channels, (3, 3), 1, padding="same")(x)
    x = act(x)
    x = tfk.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)(x)
    x = tfk.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    for _ in range(4):
        x = tfk.layers.SeparableConv2D(2 * num_channels, (3, 3), 1, padding="same")(x)
        x = act(x)
        x = tfk.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)(x)
        x = tfk.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = tfk.layers.SeparableConv2D(4 * num_channels, (3, 3), 1, padding="same")(x)
    x = act(x)
    # x = tfk.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)(x)
    x = tfk.layers.GlobalMaxPooling2D()(x)

    # x = tfk.layers.Dense(2 * num_channels)(x)
    # x = act(x)
    # x = tfk.layers.Dropout(dropout_perc)(x)

    # x = tfk.layers.Dense(num_channels)(x)
    # x = act(x)
    # x = tfk.layers.Dropout(dropout_perc)(x)

    x = tfk.layers.Dense(num_classes, activation="softmax", name="classification_head")(
        x
    )

    # model_name = "depthwise_small"

    # inputs = tfk.Input(shape=input_shape, name="inputs")
    # x = tf.image.per_image_standardization(inputs)
    # x = tfk.layers.GaussianNoise(stddev=0.01)(x)

    # x = tfk.layers.SeparableConv2D(num_channels, (3, 3), 1, padding="same")(x)
    # x = tfk.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)(x)
    # x = act(x)
    # x = tfk.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # x = tfk.layers.SeparableConv2D(2 * num_channels, (3, 3), 1, padding="same")(x)
    # x = tfk.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)(x)
    # x = act(x)
    # x = tfk.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # x = tfk.layers.SeparableConv2D(4 * num_channels, (3, 3), 1, padding="same")(x)
    # x = tfk.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)(x)
    # x = act(x)

    # x = tfk.layers.GlobalMaxPooling2D()(x)

    # x = tfk.layers.Dense(2 * num_channels)(x)
    # x = act(x)
    # x = tfk.layers.Dropout(dropout_perc)(x)

    # x = tfk.layers.Dense(num_channels)(x)
    # x = act(x)
    # x = tfk.layers.Dropout(dropout_perc)(x)

    # x = tfk.layers.Dense(num_classes, activation="softmax", name="classification_head")(
    #     x
    # )

    return tfk.models.Model(inputs=inputs, outputs=[x], name=model_name)


def build_conv_small(
    input_shape, act, dropout_perc, num_classes, num_channels, momentum, epsilon
):

    model_name = "conv_small"

    inputs = tfk.Input(shape=input_shape, name="inputs")
    x = tf.image.per_image_standardization(inputs)
    x = tfk.layers.GaussianNoise(stddev=0.01)(x)

    x = tfk.layers.Conv2D(num_channels, (3, 3), 1, padding="same")(x)
    x = tfk.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)(x)
    x = act(x)
    x = tfk.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    for _ in range(4):
        x = tfk.layers.Conv2D(2 * num_channels, (3, 3), 1, padding="same")(x)
        x = tfk.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)(x)
        x = act(x)
        x = tfk.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = tfk.layers.Conv2D(4 * num_channels, (3, 3), 1, padding="same")(x)
    x = tfk.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)(x)
    x = act(x)
    x = tfk.layers.GlobalMaxPooling2D()(x)

    # x = tfk.layers.Dense(2 * num_channels)(x)
    # x = act(x)
    # x = tfk.layers.Dropout(dropout_perc)(x)

    # x = tfk.layers.Dense(num_channels)(x)
    # x = act(x)
    # x = tfk.layers.Dropout(dropout_perc)(x)

    x = tfk.layers.Dense(num_classes, activation="softmax", name="classification_head")(
        x
    )

    return tfk.models.Model(inputs=inputs, outputs=[x], name=model_name)
