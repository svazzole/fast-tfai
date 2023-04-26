import tensorflow as tf
import tensorflow.keras as tfk


# TODO: check pooling <-> batchnorm. Which goes before the other?
#       Ans: batchnorm after pooling
def build_custom_cnn(
    input_shape, momentum, epsilon, act, dropout_perc, num_channels, num_classes
):

    model_name = "clf_custom_cnn"

    inputs = tfk.Input(shape=input_shape, name="inputs")
    x = tf.image.per_image_standardization(inputs)
    x = tfk.layers.GaussianNoise(stddev=0.01)(x)

    x = tfk.layers.Conv2D(num_channels, (3, 3), 1, padding="same")(x)
    x = tfk.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)(x)
    x = act(x)
    x = tfk.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = tfk.layers.Conv2D(2 * num_channels, (3, 3), 1, padding="same")(x)
    x = tfk.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)(x)
    x = act(x)
    x = tfk.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = tfk.layers.Conv2D(8 * num_channels, (3, 3), 1, padding="same")(x)
    x = tfk.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)(x)
    x = act(x)
    x = tfk.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = tfk.layers.Conv2D(
        16 * num_channels, (3, 3), 1, padding="same", name="last_conv"
    )(x)
    x = tfk.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)(x)
    x = act(x)

    x = tfk.layers.GlobalMaxPooling2D()(x)

    x = tfk.layers.Dense(4 * num_channels)(x)
    x = act(x)
    x = tfk.layers.Dropout(dropout_perc)(x)

    x = tfk.layers.Dense(num_channels)(x)
    x = act(x)
    x = tfk.layers.Dropout(dropout_perc)(x)

    x = tfk.layers.Dense(num_classes, activation="softmax", name="classification_head")(
        x
    )

    model = tfk.models.Model(inputs=inputs, outputs=[x], name=model_name)
    return model
