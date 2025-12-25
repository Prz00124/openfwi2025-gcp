#!pip install tensorflow-tpu -f https://storage.googleapis.com/libtpu-tf-releases/index.html --force-reinstall

import os

os.environ["TF_DEVICE_USE_XLA_PJRT"] = "1" # pjrt runtime instance required
TPU_NAME = ""
os.environ["TPU_NAME"] = TPU_NAME


import numpy as np
import tensorflow as tf
import keras

print(tf.__version__, keras.__version__)


# Configs
### strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # tpu name, local for tpu vm
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    
except:
    strategy = tf.distribute.get_strategy()

tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
FORWARD_DTYPE = tf.bfloat16
print()

### hardware info
NUM_DEVICE = strategy.num_replicas_in_sync
UNIT_GB = 16

DEV_NUM_DEVICE = 1
DEV_UNIT_GB = 16

### training
CONTINUE = False
TARGET_EPOCHS = 100
DO_EPOCHS = 100

INIT_CKPT_DIR = "gs://model_v6_ckpt"
TRAIN_CKPT_DIR = "gs://model_v6_ckpt"

BATCH_SIZE = 8
WARMUP_EPOCHS = 1
LR_INIT = 1e-4
LR_END = 1e-6
WEIGHT_DECAY = 1e-1

### preprocessing

### model

### dynamic adjustment
scaling_factor = NUM_DEVICE / DEV_NUM_DEVICE * UNIT_GB / DEV_UNIT_GB
BATCH_SIZE = int(BATCH_SIZE* scaling_factor)
LR_INIT = LR_INIT* scaling_factor

print("BATCH_SIZE: ", BATCH_SIZE, "  |  LR_INIT: ", LR_INIT)


# IO
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

# Model

# %% [markdown]
# ### Encoders

# %%
@keras.saving.register_keras_serializable()
class bSiLU(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(bSiLU, self).__init__(**kwargs)

    def build(self, input_shapes):
        b_shape = [1] * len(input_shapes)
        b_shape[-1] = input_shapes[-1]
        self.b = self.add_weight(
            shape=b_shape,
            initializer=tf.keras.initializers.Constant(1.702),
            dtype=tf.float32,
            trainable=True,
            name="beta",
        )

    def call(self, inputs):
        return tf.nn.silu(inputs, self.b)

# %%
### residual conv block
@keras.saving.register_keras_serializable()
class ResConvBlock(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(ResConvBlock, self).__init__(**kwargs)

    def build(self, input_shapes):
        self.conv1 = tf.keras.layers.Conv2D(filters=input_shapes[-1] // 2,
                                            kernel_size=3,
                                            strides=1,
                                            padding='same',
                                            activation="relu")
        self.conv2 = tf.keras.layers.Conv2D(filters=input_shapes[-1] // 2,
                                            kernel_size=3,
                                            strides=1,
                                            padding='same',
                                            activation="relu")
        self.proj = tf.keras.layers.Dense(input_shapes[-1])

    def call(self, inputs):
        return inputs + self.proj(self.conv2(self.conv1(inputs)))

def build_model():
    # Inputs
    Input_x = tf.keras.Input((72, 72, 3), name = "x")

    return tf.keras.Model(Input_x, y)


# %%
@keras.saving.register_keras_serializable()
class EMA_MAE_Loss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.98, beta = 1, name="ema_mae"):
        super().__init__(name=name)
        self.alpha = alpha
        self.epsilon = 1e-3
        self.beta = beta
        self.ema_mae = tf.Variable(1.0, trainable=False, dtype=tf.float32)

    def call(self, y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        mae = tf.reduce_mean(tf.abs(y_true - y_pred))

        new_ema = self.alpha * self.ema_mae + (1.0 - self.alpha) * mae
        self.ema_mae.assign(new_ema)
        
        ema_value = tf.stop_gradient(self.ema_mae)

        loss = mse / (ema_value + self.epsilon)

        return self.beta* loss
        
    def get_config(self):
        base_config = super().get_config()
        config = {
            "alpha": self.alpha,
            "beta":self.beta
            }
        return {**base_config, **config}

with strategy.scope():
    m = build_model()

m.summary()


### Loss, Optimizer, Compiling and Checkpointing
with strategy.scope():
    LR_SCHE = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-6,
        decay_steps=int(train_card * TARGET_EPOCHS),
        alpha=LR_END / LR_INIT,
        name='CosineDecay',
        warmup_target=LR_INIT,
        warmup_steps=int(train_card * WARMUP_EPOCHS),
    )
    opt = tf.keras.optimizers.AdamW(  #Lion
        learning_rate=LR_SCHE,
        weight_decay=WEIGHT_DECAY,
        beta_1=0.9,
        beta_2=0.98,
        clipnorm=1,
    )
    m.compile(
        optimizer=opt,
        loss=EMA_MAE_Loss(),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.MeanSquaredError(name='mse')
        ],
        run_eagerly=False,
        jit_compile=True,
    )

    epoch_counter = tf.Variable(0, dtype=tf.int32, trainable=False)

# %%
with strategy.scope():
    checkpoint = tf.train.Checkpoint(model=m,
                                        optimizer=opt,
                                        epoch=epoch_counter)
    train_manager = tf.train.CheckpointManager(checkpoint,
                                                directory=TRAIN_CKPT_DIR,
                                                max_to_keep=None)

    if CONTINUE:
        restore_manager = tf.train.CheckpointManager(
            checkpoint, directory=INIT_CKPT_DIR, max_to_keep=None)
        checkpoint.restore(
            restore_manager.latest_checkpoint)  #.expect_partial()
        print(
            f"Restored from {restore_manager.latest_checkpoint}, starting at epoch {int(epoch_counter.numpy())}"
        )
    else:
        train_manager.save()


# %%
tra_mae = []
val_mae = []

for i in range(DO_EPOCHS):
    print(f"EPOCH {int(epoch_counter.numpy())}")
    results = m.fit(
        train_set,
        steps_per_epoch= train_card,
        verbose=2,
        validation_data=valid_set,
        validation_steps= valid_card
    )
    epoch_counter.assign_add(1)
    train_manager.save()

    tra_mae += results.history["val_mae"]
    val_mae += results.history["mae"]
