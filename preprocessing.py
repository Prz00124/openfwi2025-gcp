import os
import argparse

import tensorflow as tf
import numpy as np

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_example(seis_data, velo_data, style_text):
    """
    seis_data: numpy array, shape (5, 1000, 70), dtype=float32
    velo_data: numpy array, shape (70, 70), dtype=float32
    style_text: string or bytes
    """
    feature = {
        'seis': _bytes_feature(seis_data.astype(np.float32).tobytes()),
        'velo': _bytes_feature(velo_data.astype(np.float32).tobytes()),
        'style': _bytes_feature(style_text.encode('utf-8') if isinstance(style_text, str) else style_text),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def make_tfrecord(files, num_per_tfrecord, output_folder):
    options = tf.io.TFRecordOptions(compression_type="GZIP")

    file_idx = 0
    counter = 0
    writer = None
    for subtype, seis_dir, velo_dir in files:
        if counter % num_per_tfrecord == 0:
            if writer:
                writer.close()
            file_path = os.path.join(output_folder, f"part_{file_idx:05d}.tfrecord")
            writer = tf.io.TFRecordWriter(file_path, options=options)
            file_idx += 1
        
        seis = np.load(seis_dir)
        velo = np.load(velo_dir)

        example = serialize_example(seis, velo, subtype)
        writer.write(example)
        sample_count += 1

    if writer:
        writer.close()

if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-folder', type=str)
    parser.add_argument('--output-folder', type=str)

    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder

    # IO
    subtypes = os.listdir(input_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        os.makedirs(os.path.join(output_folder, 'train'))
        os.makedirs(os.path.join(output_folder, 'valid'))

    # build file index with format
    # (subtype, seis_dir, velo_dir)
    all_files = []
    for type_ in subtypes:
        seis_folder = os.path.join(input_folder, type_, 'seismic')
        velo_folder = os.path.join(input_folder, type_, 'velocity')
        n = os.listdir(seis_folder)
        for i in range(n):
            all_files.append(
                type_,
                os.path.join(seis_folder, str(i).zfill(6) +'.npy'),
                os.path.join(velo_folder, str(i).zfill(6) +'.npy'),
            )
    # shuffling the data format
    np.random.shuffle(all_files)

    # train valid split (95:5)
    n = len(all_files)
    valid_files = all_files[:n//20]
    train_files = all_files[n//20:]

    # zip into tfrecords
    make_tfrecord(train_files, 250, os.path.join(output_folder, 'train'))
    make_tfrecord(valid_files, 250, os.path.join(output_folder, 'valid'))