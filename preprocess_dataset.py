import os
import numpy as np
import argparse
import tensorflow as tf
from tqdm import tqdm

def load_image(filename, img_size=[256,256,3]):
    image = tf.io.read_file(filename)
    image = decode_image(image, img_size)
    return image

def decode_image(image, img_size=[256,256,3]):
    image = tf.image.decode_jpeg(image, channels=3)
    image = (tf.cast(image, tf.float32) / 127.5) - 1        
    image = tf.reshape(image, img_size)        
    return image

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # Paths
    ap.add_argument('-i','--input_dir', required=False, default="data")
    ap.add_argument('-o','--output_dir', required=False, default=None)
    
    # Data parameters
    ap.add_argument('--img_w', type=int, default=256, help='The size of image width')
    ap.add_argument('--img_h', type=int, default=256, help='The size of image hegiht')
    ap.add_argument('--img_ch', type=int, default=3, help='The size of image channel')

    args = ap.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

    # Check input directory
    assert os.path.isdir(args.input_dir), "Input directory not found"

    # Check output directory
    if(args.output_dir is None):
        args.output_dir = args.input_dir + '_npy'

    if(not os.path.isdir(args.output_dir)):
        os.makedirs(args.output_dir)

    img_shape = [args.img_w, args.img_h, args.img_ch]

    # Process all images
    for f in tqdm(os.listdir(args.input_dir)):
        name, ext = os.path.splitext(f)
        image = load_image(os.path.join(args.input_dir, f))
        np.save(os.path.join(args.output_dir, name+'.npy'), image.numpy())