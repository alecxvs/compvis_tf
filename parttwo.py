if __name__ == "__main__":
    from tensorflow.python import keras
    import random
    import os
    import pathlib
    import pandas

    # log_dir = '/tmp/tensorflow_logdir'
    log_dir = 'C:\\tmp\\tensorflow_logdir'
    num_cat = 15
    num_img = 700

    training_size = int(num_img/8*6)
    test_size = int(num_img/8)
    validation_size = num_img - training_size - test_size

    pd_columns = ["filename", "class"]

    train_images_labels = pandas.DataFrame(columns=pd_columns)
    val_images_labels = pandas.DataFrame(columns=pd_columns)
    test_images_labels = pandas.DataFrame(columns=pd_columns)

    print("Gathering images...")
    oddset = 0
    label_path: pathlib.PosixPath
    for i, label_path in enumerate(pathlib.Path("./NWPU-RESISC45").iterdir()):
        if i >= num_cat:
            break
        if not label_path.is_dir():
            oddset += 1
            continue
        i -= oddset

        images = list(map(lambda p: str(p.resolve()), label_path.glob("./*.jpg")))

        train_images_labels = train_images_labels.append(pandas.DataFrame({"filename": images[:training_size], "class": [label_path.name]*training_size}), ignore_index=True)

        val_images_labels = val_images_labels.append(pandas.DataFrame({"filename": images[training_size:-test_size], "class": [label_path.name]*validation_size}), ignore_index=True)

        test_images_labels = test_images_labels.append(pandas.DataFrame({"filename": images[-test_size:], "class": [label_path.name]*test_size}), ignore_index=True)


    print("Splitting datasets...")
    img_gen = keras.preprocessing.image.ImageDataGenerator()

    training_gen = img_gen.flow_from_dataframe(
        train_images_labels,
        batch_size=16,
        class_mode="sparse"
    )

    validation_gen = img_gen.flow_from_dataframe(
        val_images_labels,
        batch_size=1,
        class_mode="sparse"
    )

    testing_gen = img_gen.flow_from_dataframe(
        test_images_labels,
        batch_size=1,
        class_mode="sparse"
    )

    conv_args = {
        "kernel_size": 3,
        "bias_initializer": "zeros",
        "bias_regularizer": keras.regularizers.l2(0.001),
        "kernel_initializer": "random_uniform",
        "kernel_regularizer": keras.regularizers.l1(0.001)
    }

    print("Building network model...")
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, **conv_args, input_shape=(256,256,3)))
    model.add(keras.layers.MaxPooling2D())
    model.add(keras.layers.Conv2D(32, **conv_args))
    model.add(keras.layers.MaxPooling2D())
    model.add(keras.layers.Conv2D(32, **conv_args))
    model.add(keras.layers.MaxPooling2D())
    model.add(keras.layers.Conv2D(32, **conv_args))
    model.add(keras.layers.MaxPooling2D())
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(4096, bias_initializer="zeros", kernel_initializer="random_uniform"))
    # model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(4096, bias_initializer="zeros", kernel_initializer="random_uniform"))
    # model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(num_cat, activation="softmax"))
    model.compile(keras.optimizers.Adam(amsgrad=True), loss="sparse_categorical_crossentropy", metrics=['accuracy'])

    model.summary()

    tb_cb = keras.callbacks.TensorBoard(
        log_dir,
        histogram_freq=1,
        update_freq="batch"
    )

    print("Running...")
    model.fit_generator(
        training_gen,
        steps_per_epoch=80,
        epochs=50,
        validation_data=validation_gen,
        verbose=2,
        workers=8,
        callbacks=[tb_cb]
        # use_multiprocessing=True
    )

