if __name__ == "__main__":
    from tensorflow.python import keras
    import tensorflow as tf
    import numpy as np
    import random
    import os
    import platform
    import pathlib
    import pandas
    if platform.system() == "Linux":
        import tensorflow_addons as tfa

    log_dir = 'C:\\tmp\\tensorflow_logdir' if platform.system() == "Windows" else '/tmp/tensorflow_logdir'
    num_cat = 15
    num_img = 700

    training_size = int(num_img/7*5)
    test_size = int(num_img/7)
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
        class_mode="sparse",
        y_col="class"
    )

    validation_gen = img_gen.flow_from_dataframe(
        val_images_labels,
        batch_size=1,
        class_mode="sparse",
        y_col="class"
    )

    testing_gen = img_gen.flow_from_dataframe(
        test_images_labels,
        batch_size=1,
        class_mode="sparse",
        y_col="class"
    )

    conv_args = {
        "kernel_size": 3,
        "bias_initializer": "zeros",
        "bias_regularizer": keras.regularizers.l2(0.01),
        "kernel_initializer": "random_uniform",
        "kernel_regularizer": keras.regularizers.l1(0.01)
    }

    @tf.function
    def triplet_loss(labels, embeddings):
        # if tf.shape(labels).shape != 1:
        # print(tf.shape(labels))
        # print(labels)
        
        return tfa.losses.triplet_semihard_loss(tf.math.argmax(labels, axis=1), embeddings)

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
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(4096, bias_initializer="zeros", kernel_initializer="random_uniform"))
    # model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(num_cat, activation="softmax"))
    # model.add(keras.layers.Lambda(tensor_argmax))
    model.compile(keras.optimizers.Adam(amsgrad=True), loss=triplet_loss, metrics=['accuracy'])

    model.summary()

    tb_cb = keras.callbacks.TensorBoard(
        log_dir,
        histogram_freq=1,
        update_freq="batch"
    )

    print("Running...")
    model.fit_generator(
        training_gen,
        steps_per_epoch=40,
        epochs=20,
        validation_data=validation_gen,
        verbose=2,
        workers=16,
        callbacks=[tb_cb]
        # use_multiprocessing=True
    )

    model.evaluate_generator(
        testing_gen,
        verbose=2,
        workers=8,
        callbacks=[tb_cb]
    )
