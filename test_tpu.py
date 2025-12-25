#TPU_NAME = "local" # single vm
TPU_NAME = "test123" # multi vm, that is for pod (slice).

import os
import tensorflow as tf

print("Tensorflow version " + tf.__version__)
print("Detect device: ", tf.config.list_physical_devices())# this won't work while using pod (slice) because in pod slice.

cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(TPU_NAME)
tf.config.experimental_connect_to_cluster(cluster_resolver)
tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)

#strategy = tf.distribute.get_strategy() didn't work on tpu.

@tf.function
def add_fn(x,y):
  z = x + y
  return z

x = tf.constant(1.)
y = tf.constant(1.)
z = strategy.run(add_fn, args=(x, y))
print(z)