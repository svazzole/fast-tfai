import tensorflow as tf
import tensorflow.keras as tfk

from fast_tfai.models.model_builder import ModelBuilder


@ModelBuilder.register("test")
def build_simple_model(
    input_shape=(224, 224, 3), num_classes=2, num_channels=32, **kwargs
):

    model_name = "simple_model"
    act = tfk.layers.Activation("relu")
    momentum = 0.9
    epsilon = 0.001

    inputs = tfk.Input(shape=input_shape, name="inputs")
    x = tf.image.per_image_standardization(inputs)
    x = tfk.layers.GaussianNoise(stddev=0.01)(x)

    x = tfk.layers.Conv2D(num_channels, (3, 3), 1, padding="same")(x)
    x = tfk.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)(x)
    x = act(x)
    x = tfk.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = tfk.layers.Conv2D(
        16 * num_channels, (3, 3), 1, padding="same", name="last_conv"
    )(x)
    x = tfk.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)(x)
    x = act(x)

    x = tfk.layers.GlobalMaxPooling2D()(x)

    x = tfk.layers.Dense(num_classes, activation="softmax", name="classification_head")(
        x
    )

    model = tfk.models.Model(inputs=inputs, outputs=[x], name=model_name)
    return model


if __name__ == "__main__":
    from fast_tfai.trainer.train import train

    train()
