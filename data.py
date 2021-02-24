import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

def data_augment(image, img_size=[256,256,3]):
    # Random cropping
    p_crop = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    if p_crop > .5:
        image = tf.image.resize(image, [286, 286])
        image = tf.image.random_crop(image, size=img_size)
        if p_crop > .9:
            image = tf.image.resize(image, [300, 300])
            image = tf.image.random_crop(image, size=img_size)
    
    # Random rotation
    p_rotate = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    if p_rotate > .9: # rotate 270 deg
        image = tf.image.rot90(image, k=3) 
    elif p_rotate > .7: # rotate 180 deg
        image = tf.image.rot90(image, k=2) 
    elif p_rotate > .5: # rotate 90 deg
        image = tf.image.rot90(image, k=1) 
    
    # Random mirroring
    p_spatial = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    if p_spatial > .6:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        if p_spatial > .9:
            image = tf.image.transpose(image)

    return image

def decode_image(image, img_size=[256,256,3]):
    image = tf.image.decode_jpeg(image, channels=3)
    image = (tf.cast(image, tf.float32) / 127.5) - 1        
    image = tf.reshape(image, img_size)             
    return image

def get_dataset(dataset_dir, augment=None, repeat=True, shuffle=True, batch_size=1, autotune=tf.data.experimental.AUTOTUNE, cache=True, from_npy=False):
    if from_npy:
        monet_filenames = tf.io.gfile.glob(str(os.path.join(dataset_dir, 'monet_npy', '*.npy')))
        photo_filenames = tf.io.gfile.glob(str(os.path.join(dataset_dir, 'photo_npy', '*.npy')))

        monet_ds = load_dataset_from_npy(monet_filenames)
        photo_ds = load_dataset_from_npy(photo_filenames)

        n_monet_samples = len(os.listdir(os.path.join(dataset_dir, 'monet_npy')))
        n_photo_samples = len(os.listdir(os.path.join(dataset_dir, 'photo_npy')))

    else:
        monet_filenames = tf.io.gfile.glob(str(os.path.join(dataset_dir, 'monet_tfrec', '*.tfrec')))
        photo_filenames = tf.io.gfile.glob(str(os.path.join(dataset_dir, 'photo_tfrec', '*.tfrec')))

        monet_ds = load_dataset(monet_filenames)
        photo_ds = load_dataset(photo_filenames)

        n_monet_samples = count_data_items(monet_filenames)
        n_photo_samples = count_data_items(photo_filenames)

    if cache:
        monet_ds = monet_ds.cache()
        photo_ds = photo_ds.cache()
        
    if augment:
        monet_ds = monet_ds.map(augment, num_parallel_calls=autotune)
        photo_ds = photo_ds.map(augment, num_parallel_calls=autotune)

    monet_ds = monet_ds.batch(batch_size, drop_remainder=True)
    photo_ds = photo_ds.batch(batch_size, drop_remainder=True)

    if repeat:
        monet_ds = monet_ds.repeat()
        photo_ds = photo_ds.repeat()
        
    if shuffle and not from_npy:
        monet_ds = monet_ds.shuffle(2048)
        photo_ds = photo_ds.shuffle(2048)
    
    # monet_ds = monet_ds.prefetch(autotune)
    # photo_ds = photo_ds.prefetch(autotune)
    
    gan_ds = tf.data.Dataset.zip((monet_ds, photo_ds))
    return gan_ds, n_monet_samples, n_photo_samples

def get_default_dataset(monet_filenames, photo_filenames, autotune=tf.data.experimental.AUTOTUNE):
    dataset = get_dataset(
        monet_filenames, 
        photo_filenames, 
        augment=False, 
        repeat=True, 
        shuffle=True, 
        batch_size=1,
        autotune=autotune
    )
    return dataset

# TODO: pass the autotune parameter
def load_dataset(filenames, labeled=True, ordered=False, autotune=tf.data.experimental.AUTOTUNE):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=autotune)
    return dataset

def load_dataset_from_npy(filenames, labeled=True, ordered=False, autotune=tf.data.experimental.AUTOTUNE):
    def load_npy(filename):
        return tf.cast(np.load(filename.numpy()), dtype=tf.float32)

    def process_path(filename):
        arr = tf.py_function(load_npy, inp=[filename], Tout=[tf.float32])
        return arr[0]
    
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.map(process_path)
    return dataset

def read_tfrecord(example):
    tfrecord_format = {
        "image_name": tf.io.FixedLenFeature([], tf.string),
        "image":      tf.io.FixedLenFeature([], tf.string),
        "target":     tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, tfrecord_format)  
    image = decode_image(example['image'])    
    return image

def visualize_images(dataset, rows=1, cols=5):
    ds_iter = iter(dataset)
    fig = plt.figure(figsize=(25, rows*5.05))
    for i in range(cols*rows):
        image = next(ds_iter)
        image = image.numpy()
        ax = fig.add_subplot(rows, cols, i+1, xticks=[], yticks=[])
        ax.imshow(image[0] * 0.5 + .5)
        plt.show()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=False, default='data')
    args = ap.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

    monet_filenames = tf.io.gfile.glob(str(os.path.join(args.dataset, 'monet_tfrec', '*.tfrec')))
    photo_filenames = tf.io.gfile.glob(str(os.path.join(args.dataset, 'photo_tfrec', '*.tfrec')))

    n_monet_samples = count_data_items(monet_filenames)
    n_photo_samples = count_data_items(photo_filenames)

    print('Number of Monet TFRecord Files:', len(monet_filenames))
    print('Number of Photo TFRecord Files:', len(photo_filenames))

    monet_filenames = tf.io.gfile.glob(str(os.path.join(args.dataset, 'monet_npy', '*.npy')))
    photo_filenames = tf.io.gfile.glob(str(os.path.join(args.dataset, 'photo_npy', '*.npy')))

    monet_dataset = load_dataset_from_npy(monet_filenames, labeled=True).batch(8, drop_remainder=True).repeat()
    photo_dataset = load_dataset_from_npy(photo_filenames, labeled=True).batch(8, drop_remainder=True).repeat()

    example_monet = next(iter(monet_dataset))
    example_photo = next(iter(photo_dataset))

    print(example_monet.shape, example_photo.shape)

    # visualize_images(monet_dataset, rows=2, cols=4)
    # visualize_images(photo_dataset, rows=2, cols=4)

    dataset, n_monet_samples, n_photo_samples = get_dataset(
        dataset_dir=args.dataset,
        augment=False, 
        repeat=True, 
        shuffle=True, 
        batch_size=8,
        from_npy=True,
        cache=False
    )

    example_monet, example_photo  = next(iter(dataset))
    print(example_monet.shape, example_photo.shape)