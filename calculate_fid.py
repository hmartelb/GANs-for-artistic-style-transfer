import argparse
import os
from pathlib import Path

import numpy as np
import skimage
from PIL import Image
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import (InceptionV3,
                                                        preprocess_input)
from tensorflow.keras.datasets import cifar10


# Calculate frechet inception distance (FID)
def calculate_fid(images1, images2, model=None, return_acts=True):
    # Prepare the inception v3 model
    if model is None:
        model = InceptionV3(include_top=False, pooling='avg', input_shape=(256,256,3))

	# Calculate activations
    act1 = model.predict(images1, verbose=1)
    act2 = model.predict(images2, verbose=1)

    # Calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    
    # Calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    
    # Calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    
    # Check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # Calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    if return_acts:
        return fid, act1, act2
    return fid

def load_image(filename, img_size=(256, 256)):
    im = Image.open(str(filename))
    return im.resize(img_size, Image.ANTIALIAS)


def load_data(data_path, img_size=(256, 256)):
    data_path = Path(data_path)
    files = list(data_path.glob('*.jpg')) + list(data_path.glob('*.png'))
    return np.array([np.array(load_image(f, img_size)) for f in files])


def distance_thresholding(d, cosine_distance_eps=0.1):
    return d if d < cosine_distance_eps else 1


def cosine_distance(features1, features2):
    features1_nozero = features1[np.sum(features1, axis=1) != 0]
    features2_nozero = features2[np.sum(features2, axis=1) != 0]
    norm_f1 = normalize_rows(features1_nozero)
    norm_f2 = normalize_rows(features2_nozero)

    d = 1.0 - np.abs(np.matmul(norm_f1, norm_f2.T))
    mean_min_d = np.mean(np.min(d, axis=1))
    return mean_min_d


def normalize_rows(x):
    return np.nan_to_num(x/np.linalg.norm(x, ord=2, axis=1, keepdims=True))


def evaluate(images1, images2, fid_epsilon=1e-14, cosine_distance_eps=0.1):
    fid_value, act1, act2 = calculate_fid(images1, images2, return_acts=True)
    d = cosine_distance(act1, act2)
    d = distance_thresholding(d, cosine_distance_eps)
    return fid_value, d, fid_value/(d + fid_epsilon)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-i','--original', required=False, default=os.path.join('data', 'monet_jpg'))
    ap.add_argument('-o','--generated', required=False, default=os.path.join('data', 'monet_jpg'))
    ap.add_argument('--gpu', required=False, default='0')
    args = ap.parse_args()

    assert os.path.isdir(args.original), 'Input data not found'
    assert os.path.isdir(args.generated), 'Generated data not found'

    # Select which GPU to use and enable mixed precision
    print('Using GPU: '+ args.gpu)
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

    monet_original = preprocess_input(load_data(args.original))
    monet_generated = preprocess_input(load_data(args.generated))

    print(monet_original.shape, monet_generated.shape)
    
    fid_value, distance, mi_fid_score = evaluate(monet_original, monet_generated)
    
    print(f'FID: {fid_value:.5f}')
    print(f'distance: {distance:.5f}')
    print(f'MiFID: {mi_fid_score:.5f}')
