from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import glob
import json
import math
import os
import random
import re
import shutil
import time
from datetime import datetime

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import seaborn as sns
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm

from callbacks import CustomModelCheckpoint, GenerateSamples, CustomLearningRateSchedule
from data import count_data_items, get_dataset
from models import cyclegan, munit_v2, ugatit


def get_callbacks(args):
    callbacks_list = []
    
    model_name = get_model_name(args)

    # Save model weights
    checkpoint = CustomModelCheckpoint(
        save_dir=os.path.join(args.checkpoint_dir, model_name),
        save_always=True
    )
    callbacks_list.append(checkpoint)

    # Save image samples
    generate_samples = GenerateSamples(
        examples_dir=os.path.join(args.examples_dir),
        save_dir=os.path.join(args.sample_dir, model_name),
        img_size=[args.img_w, args.img_h, args.img_ch]
    )
    callbacks_list.append(generate_samples)

    if(args.architecture == 'munit'):
        def scheduler(epoch, lr):
            return lr * pow(0.5, epoch)
        
        lr_scheduler = CustomLearningRateSchedule(
            scheduler, 
            init_gen_lr=args.gen_lr,
            init_disc_lr=args.disc_lr,
            verbose=1
        )
        callbacks_list.append(lr_scheduler)

    return callbacks_list
  

def get_model(args):
    if args.architecture == 'cyclegan':
        return cyclegan.get_compiled_model(args)
    elif args.architecture == 'munit':
        return munit_v2.get_compiled_model(args)
    elif args.architecture == 'ugatit':
        return ugatit.get_compiled_model(args)
    return None


def get_model_name(args):
    model_name = f"{args.architecture}_{args.gan_type}_{args.base_channels}"
    if args.sn:
        model_name += "_sn"
    return model_name


def generate(args):
    # # Get the dataset
    dataset, n_monet_samples, n_photo_samples = get_dataset(
        args.dataset,
        augment=args.augment, 
        repeat=True, 
        shuffle=False, 
        from_npy=args.from_npy,
        batch_size=1
    )
    dataset_iter = iter(dataset)

    # # Get the model and restore the checkpoint
    model_name = get_model_name(args)
    model = get_model(args)
    model.load(os.path.join(args.checkpoint_dir, model_name))
    # model.load(filepath=os.path.join(args.checkpoint_dir, model_name, "model_name.h5"))

    out_dir = os.path.join(args.result_dir, model_name)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    for i in tqdm(range(n_photo_samples)):
        # Get the image from the dataset iterator
        style_ref, img = next(dataset_iter)

        # Get a prediction and save
        # if args.architecture == 'munit':
        #     prediction = model.generate_guide(img, style_ref)
        # else:
        prediction = model.generate(img)
        prediction = tf.squeeze(prediction).numpy()
        prediction = (prediction * 127.5 + 127.5).astype(np.uint8)   
        out_img = PIL.Image.fromarray(prediction)
        out_img.save(os.path.join(out_dir, str(i).zfill(4)+'.jpg'))


def train(args):
    # Get the dataset
    dataset, n_monet_samples, n_photo_samples = get_dataset(
        args.dataset,
        augment=args.augment, 
        repeat=True, 
        shuffle=True, 
        batch_size=args.batch_size,
        autotune=1,
        from_npy=args.from_npy,
        cache=False
    )

    # Get the model
    model = get_model(args)
    
    # Try loading pretrained weights
    model_name = get_model_name(args)
    # try:
    #     model.load(filepath=os.path.join(args.checkpoint_dir, model_name))
    #     print("Model weights restored.")
    # except:
    #     print("Could not find model weights.")
    
    # Train the model
    history = model.fit(
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        steps_per_epoch=(max(n_monet_samples, n_photo_samples)//args.batch_size),
        # steps_per_epoch=1000,
        callbacks=get_callbacks(args)
    )

    # Generate the results
    # generate(args)

    # Save training history
    # history_filename = os.path.join(args.result_dir, f"{args.architecture}_{args.gan_type}_{args.epochs}.json")
    # with open(history_filename, 'w') as f:
    #     json.dump(history.history, f)



if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    # Execution parameters
    ap.add_argument('--task', type=str, default='train', help='Choose task [train / generate / evaluate]')

    # Model parameters
    ap.add_argument('--architecture', type=str, default='cyclegan', help='GAN architecture [cyclegan / munit / ugatit]')
    ap.add_argument('--gan_type', type=str, default='lsgan', help='GAN loss type [gan / lsgan]')
    ap.add_argument('--gan_w', type=float, default=1.0, help='Weight of adversarial loss')
    ap.add_argument('--recon_x_w', type=float, default=10.0, help='Weight of image reconstruction loss')
    ap.add_argument('--recon_s_w', type=float, default=1.0, help='Weight of style reconstruction loss')
    ap.add_argument('--recon_c_w', type=float, default=1.0, help='Weight of content reconstruction loss')
    ap.add_argument('--recon_x_cyc_w', type=float, default=0.0, help='Weight of explicit style augmented cycle consistency loss')

    # UGATIT
    ap.add_argument('--cycle_weight', type=int, default=10, help='Weight Cycle')
    ap.add_argument('--identity_weight', type=int, default=10, help='Weight Identity')
    ap.add_argument('--cam_weight', type=int, default=1000, help='Weight CAM')
    ap.add_argument('--smoothing', type=bool, default=True, help='AdaLIN smoothing effect')
    ap.add_argument('--sn', type=bool, default=False, help='Use Spectral Normalization in the Discriminator')

    ap.add_argument('--base_channels', type=int, default=64, help='Base channel number per layer')
    ap.add_argument('--style_dim', type=int, default=8, help='Length of style code')
    ap.add_argument('--n_sample', type=int, default=2, help='Number of sampling layers in content encoder')
    ap.add_argument('--n_res', type=int, default=4, help='Number of residual blocks in content encoder/decoder')

    ap.add_argument('--n_dis', type=int, default=4, help='Number of discriminator layer')
    ap.add_argument('--n_scale', type=int, default=3, help='Number of scales')

    # Data parameters
    ap.add_argument('--img_w', type=int, default=256, help='Input image width')
    ap.add_argument('--img_h', type=int, default=256, help='Input image hegiht')
    ap.add_argument('--img_ch', type=int, default=3, help='Input image numnber of channels (RGB = 3)')
    ap.add_argument('--augment', type=bool, default=False, help='Use data augmentation or not (True / False)')
    
    # Paths
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--from_npy', type=bool, required=False, default=False)
    ap.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory name to save the model checkpoints')
    ap.add_argument('--result_dir', type=str, default='results', help='Directory name to save the generated images')
    ap.add_argument('--examples_dir', type=str, default='examples', help='Directory from which to load the samples on training')
    ap.add_argument('--sample_dir', type=str, default='samples', help='Directory name to save the samples on training')
    ap.add_argument('--style_reference', type=str, default=os.path.join('monet_jpg','1f22663e72.jpg'), help='Monet painting to be used as style reference in guided translation')
    # ap.add_argument('--log_dir', type=str, default='logs', help='Directory name to save training logs')

    # Training parameters
    ap.add_argument('--epochs', required=False, default=100, type=int)
    ap.add_argument('--batch_size', required=False, default=1, type=int)
    ap.add_argument('--gen_lr', required=False, default=1e-4, type=float)
    ap.add_argument('--disc_lr', required=False, default=1e-4, type=float)

    # Hardware settings
    ap.add_argument('--gpu', required=False, default='0')

    args = ap.parse_args()

    # Select which GPU to use and enable mixed precision
    print('Using GPU: '+ args.gpu)
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

    if args.task == 'train':
        train(args)
    elif args.task == 'generate':
        generate(args)
    # elif args.task == 'evaluate':
    #     pass
    # else:
    #     pass
