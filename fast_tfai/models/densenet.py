import tensorflow.keras as tfk


def build_densenet121(
    input_shape=(224, 224, 3), num_classes=2, version=None, multilabel=False, **kwargs
) -> tfk.Model:

    model_name = "densenet121_classifier"
    if version:
        model_name = version
    input_tensor = tfk.layers.InputLayer(input_shape=input_shape, name="inputs")

    backbone = tfk.applications.DenseNet121(
        input_shape=input_shape, weights="imagenet", include_top=False, pooling="avg"
    )
    backbone.trainable = False
    preprocess = tfk.applications.densenet.preprocess_input

    inputs = input_tensor.input
    inputs = preprocess(inputs)
    x = backbone(inputs)

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


if __name__ == "__main__":
    # backbone = tfk.applications.DenseNet121(
    #     input_shape=(224, 224, 3), include_top=False, pooling="max"
    # )
    # backbone.summary()

    model = build_densenet121((512, 512, 3), 2)
    model.summary()
