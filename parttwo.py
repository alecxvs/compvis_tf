if __name__ == "__main__":
   from tensorflow.python import keras
   import random
   import os
   import pathlib
   import pandas

   log_dir = '/tmp/tensorflow_logdir'
   num_cat = 15
   num_img = 700

   training_size = int(num_img/10*8)
   validation_size = int(num_img/10)
   test_size = num_img - training_size - validation_size

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

      val_images_labels = val_images_labels.append(pandas.DataFrame({"filename": images[training_size:-validation_size], "class": [label_path.name]*validation_size}), ignore_index=True)

      test_images_labels = test_images_labels.append(pandas.DataFrame({"filename": images[-validation_size:], "class": [label_path.name]*test_size}), ignore_index=True)


   print("Splitting datasets...")
   img_gen = keras.preprocessing.image.ImageDataGenerator()

   training_gen = img_gen.flow_from_dataframe(
      train_images_labels,
      class_mode="sparse"
   )

   validation_gen = img_gen.flow_from_dataframe(
      val_images_labels,
      class_mode="sparse"
   )

   testing_gen = img_gen.flow_from_dataframe(
      test_images_labels,
      class_mode="sparse"
   )

   # training_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
   # validation_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
   # test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
   # with tf.Session() as sess:

   print("Building network model...")
   model = keras.Sequential()
   model.add(keras.layers.Conv2D(32, [3,3], input_shape=(256,256,3)))
   model.add(keras.layers.MaxPooling2D())
   model.add(keras.layers.Conv2D(32, [3,3]))
   model.add(keras.layers.MaxPooling2D())
   model.add(keras.layers.Conv2D(32, [3,3]))
   model.add(keras.layers.MaxPooling2D())
   model.add(keras.layers.Conv2D(32, [3,3]))
   model.add(keras.layers.Flatten())
   model.add(keras.layers.Dense(4096))
   model.add(keras.layers.Dense(4096))
   model.add(keras.layers.Softmax(input_shape=[num_cat]))
   model.compile("adadelta", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

   # model.summary()


   print("Running...")
   model.fit_generator(
      training_gen,
      steps_per_epoch=8,
      epochs=int(num_cat * num_img / 32),
      validation_data=validation_gen,
      verbose=2,
      workers=8
      # use_multiprocessing=True
   )

