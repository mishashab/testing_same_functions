import tensorflow


def make_model_easy(num_classes, img_size):
    resize_and_rescale = tensorflow.keras.Sequential(
        [
            tensorflow.keras.layers.Resizing(img_size, img_size),
            tensorflow.keras.layers.Rescaling(1.0 / 255),
        ]
    )

    inputs = tensorflow.keras.layers.Input(shape=(None, None, 3))
    x = resize_and_rescale(inputs)
    x = tensorflow.keras.layers.Conv2D(
        256, 3, padding="same", activation="relu", input_shape=(img_size, img_size, 3)
    )(x)
    x = tensorflow.keras.layers.MaxPooling2D(3, 3)(x)

    x = tensorflow.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tensorflow.keras.layers.MaxPooling2D(2, 2)(x)
    x = tensorflow.keras.layers.Dropout(0.25)(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)

    x = tensorflow.keras.layers.MaxPooling2D(3, 3)(x)
    x = tensorflow.keras.layers.Conv2D(64, 3, padding="same",
                                       activation="relu")(x)
    x = tensorflow.keras.layers.MaxPooling2D(2, 2)(x)

    x = tensorflow.keras.layers.Dropout(0.25)(x)
    x = tensorflow.keras.layers.Conv2D(32, 3, padding="same",
                                       activation="relu")(x)
    x = tensorflow.keras.layers.MaxPooling2D(2, 2)(x)

    x = tensorflow.keras.layers.Dropout(0.25)(x)

    x = tensorflow.keras.layers.BatchNormalization()(x)

    x = tensorflow.keras.layers.Flatten()(x)
    x = tensorflow.keras.layers.Dense(128, activation="relu")(x)
    outputs = tensorflow.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tensorflow.keras.Model(inputs, outputs)
