import tensorflow as tf
import cv2
import numpy as np

def augment(image, label):
    image = tf.image.random_brightness(image, 0.2)
#     if tf.random.uniform([]) > 0.9:
#         image = tf.image.random_hue(image, 0.3)
    if tf.random.uniform([]) < 0.2:
        image = tf.image.random_contrast(image, 0.4, 0.6)
    if tf.random.uniform([]) > 0.5:
        image = tf.keras.backend.random_binomial(
            (image.shape[0], image.shape[1], 1), p=0.97) * image
        image = 255. - (tf.keras.backend.random_binomial(
            (image.shape[0], image.shape[1], 1), p=0.97) * (255 - image))
    if tf.random.uniform([]) > 0.5:
        image = tf.image.random_saturation(image,0.5,1)
    return image, label


def _parse_function(proto):
    keys_to_features = {'image': tf.io.FixedLenFeature([], tf.string),
                        'mask': tf.io.FixedLenFeature([], tf.string),
                        'edge_map_1': tf.io.FixedLenFeature([], tf.string),
                        'ID': tf.io.FixedLenFeature([], tf.string)}
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)
    edge_map_1 = tf.io.decode_jpeg(parsed_features['edge_map_1'])
    edge_map_1 = tf.cast(tf.reshape(edge_map_1, (512, 256)), tf.int32)
    idx = parsed_features['ID']
    image = tf.io.decode_jpeg(parsed_features['image'])
    mask = tf.io.decode_jpeg(parsed_features['mask'])
    image = tf.cast(tf.reshape(image, (512, 256, 3)), tf.float32)
    mask = tf.cast(tf.reshape(mask, (512, 256)), tf.int32)
    # mask = tf.clip_by_value(mask, clip_value_min=0, clip_value_max=6)
    mask = tf.stack([mask, edge_map_1], axis= -1)
    return image, mask, idx


def _filter(image, label, idx):
    return image, label

def _normalize(image, label, idx = None):
    image = image / 127.5 - 1
    if idx  is not None:
        return image, label, idx  
    else:
        return image, label


def create_dataset(opts, string, per_replica_batch_size =  None):
    if opts.tfrecord:
        dataset = tf.data.TFRecordDataset(opts.tfrecord_path)
        # Following 80-20 train-test split

        if string == 'train':
            k = 0.8
            dataset = dataset.take(int(k * opts.num_data))
            dataset = dataset.shuffle(2048, reshuffle_each_iteration=True)
            if per_replica_batch_size:
                batch_size = per_replica_batch_size
            else:
                batch_size = opts.train_batch_size

        elif string == 'val':
            k = 0.2
            dataset = dataset.skip(int((1-k) * opts.num_data))
            if per_replica_batch_size:
                batch_size = per_replica_batch_size
            else:
                batch_size = opts.val_batch_size
        
        elif string == 'k_worst':
            k = 1
            dataset = dataset.take(opts.num_data)
            batch_size = opts.k_worst_batch_size

        dataset = dataset.map(
            _parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if string != 'k_worst':
        dataset = dataset.map(
            _filter, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if opts.augment:
        dataset = dataset.map(
            augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            
    dataset = dataset.map( _normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size, drop_remainder=True)

    if opts.mode != 'colab_tpu':
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        setattr(dataset, 'length', int(k * opts.num_data))

    return dataset
