import os
import tensorflow as tf

BATCH_SIZE = 32
train_dir = "gs://openfwi_tfrecord"
valid_dir = "gs://openfwi_valid_tfrecords"

# %% parse
def parse_tfrecord_fn(example):
    feature_description = {
        'seis': tf.io.FixedLenFeature([], tf.string),
        'velo': tf.io.FixedLenFeature([], tf.string),
        "style": tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, feature_description)
    seis = tf.io.decode_raw(example['seis'], tf.float32)
    seis = tf.reshape(seis, [5, 1000, 70])  # 恢復原shape

    velo = tf.io.decode_raw(example['velo'], tf.float32)
    velo = tf.reshape(velo, [70, 70])
    return seis, velo


@tf.function
def preprocessing(x, y):#, sample_weights
    
    return x, y

# %%
train_set = tf.data.Dataset.list_files(os.path.join(train_dir, "*.tfrecord"), shuffle=True)
train_set = train_set.interleave(lambda filename: tf.data.TFRecordDataset(filename, compression_type='GZIP'),
                                num_parallel_calls=tf.data.AUTOTUNE,
                                deterministic=False
                                )

# %%
train_set = (train_set
.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
.shuffle(10000)
.map(preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
.batch(BATCH_SIZE, drop_remainder=True)
.prefetch(tf.data.AUTOTUNE)
)

# %%
valid_set = tf.data.Dataset.list_files(os.path.join(valid_dir, "*.tfrecord"), shuffle=True)
valid_set = valid_set.interleave(lambda filename: tf.data.TFRecordDataset(filename, compression_type='GZIP'),
                                num_parallel_calls=tf.data.AUTOTUNE,
                                deterministic=False
                                )

# %%
valid_set = (valid_set
.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
.shuffle(5000)
.map(preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
.batch(BATCH_SIZE, drop_remainder=True)
.prefetch(tf.data.AUTOTUNE)
)