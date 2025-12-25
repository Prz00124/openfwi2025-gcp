import tensorflow as tf
# ckpt
CONTINUE = False
INIT_CKPT_DIR = "gs://model_v6_ckpt"
TRAIN_CKPT_DIR = "gs://model_v6_ckpt"

# m = model
# optimizer = opt
# epoch_counter

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
