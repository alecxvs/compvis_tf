import tensorflow as tf
GRAPH_PB_PATH = './alexnet_frozen.pb'
log_dir = '/tmp/tensorflow_logdir'
with tf.Session() as sess:
   print("load graph")
   with tf.gfile.GFile(GRAPH_PB_PATH,'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
   sess.graph.as_default()
   tf.import_graph_def(graph_def, name='')

   tfgraph = sess.graph
   # pb_visual_writer = tf.summary.FileWriter(log_dir)
   # pb_visual_writer.add_graph(sess.graph)
   # print("Model Imported. Visualize by running: "
   #       "tensorboard --logdir={}".format(log_dir))

   graph_nodes=[n for n in graph_def.node]
   for t in graph_nodes:
      if t.name == "fc6/fc6":
         print(t)
      # for k,v in t.attr
      #    print("Key:", k)
      #    print("Value:", v)