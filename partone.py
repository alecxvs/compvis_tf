import tensorflow as tf
import pathlib
import pandas
import io
import pickle
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import ops
from tensorflow.python.client import session
from typing import Type, List
GRAPH_PB_PATH = './alexnet_frozen.pb'
log_dir = '/tmp/tensorflow_logdir'

print("Loading AlexNet...")
with tf.io.gfile.GFile(GRAPH_PB_PATH,'rb') as f:
   graph_def = graph_pb2.GraphDef()
   graph_def.ParseFromString(f.read())

print("Gathering images...")
num_cat = 15
num_img = 700

training_size = int(num_img/7*5)
validation_size = int(num_img/7)
test_size = num_img - training_size - validation_size

pd_columns = ["filename", "label"]

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

   train_images_labels = train_images_labels.append(pandas.DataFrame({"filename": images[:training_size], "label": [label_path.name]*training_size}), ignore_index=True)

   val_images_labels = val_images_labels.append(pandas.DataFrame({"filename": images[training_size:-validation_size], "label": [label_path.name]*validation_size}), ignore_index=True)

   test_images_labels = test_images_labels.append(pandas.DataFrame({"filename": images[-validation_size:], "label": [label_path.name]*test_size}), ignore_index=True)


training_sample: pandas.DataFrame = train_images_labels.sample(frac=1).reset_index(drop=True)
img_tensors = []

def process_image(image_path, label):
   img = tf.io.read_file(image_path)
   decoded_img = tf.io.decode_image(img, channels=3)
   resized_img = tf.image.resize_image_with_crop_or_pad(decoded_img, 227, 227)
   # return tf.reshape(resized_img, [1, 227, 227, 3]), label
   return resized_img, label

ds = tf.data.Dataset.from_tensor_slices((list(training_sample['filename']), list(training_sample['label'])))
ds = ds.map(process_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds = ds.batch(8)
ds = ds.shuffle(len(training_sample))

training_num = training_size*num_cat
trained = 0

sess = session.Session()
with sess.as_default():
   tf.graph_util.import_graph_def(graph_def, name='')

   operations: List[Type[tf.Operation]] = sess.graph.get_operations()
   with io.open("features.p", "wb") as fout:
      for img, label in iter(ds):
         feat = sess.run('fc8/fc8:0', {'Placeholder:0': img.numpy()})
         for i, l in enumerate(label):
            pickle.dump((feat[i], l), fout)
         trained += len(label)
         print(f"Processed batch of 8 ({trained} of {training_num})")

sess.close()




# pb_visual_writer = tf.summary.FileWriter(log_dir)
# pb_visual_writer.add_graph(sess.graph)
# print("Model Imported. Visualize by running: "
#       "tensorboard --logdir={}".format(log_dir))

# graph_nodes=[n for n in graph_def.node]

# for t in graph_nodes:
#    if t.name == "fc6/fc6":
#       print(t)
   # for k,v in t.attr
   #    print("Key:", k)
   #    print("Value:", v)

