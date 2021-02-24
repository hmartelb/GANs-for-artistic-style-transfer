import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
import PIL

from data import decode_image

class CustomLearningRateSchedule(Callback):
    def __init__(self, schedule, init_gen_lr, init_disc_lr, verbose=0):
        super(CustomLearningRateSchedule, self).__init__()
        self.schedule = schedule
        self.verbose = verbose

        self.init_gen_lr = init_gen_lr
        self.init_disc_lr = init_disc_lr

    def on_epoch_begin(self, epoch, logs=None):
        new_gen_lr = self.schedule(epoch, self.init_gen_lr)
        new_disc_lr = self.schedule(epoch, self.init_disc_lr)

        self.model.set_lr(new_gen_lr, new_disc_lr)

        if self.verbose > 0:
            print('\nEpoch %03d: LearningRateScheduler reducing learning rate to %s (G) and %s (D)' % (epoch + 1, new_gen_lr, new_disc_lr))

class CustomModelCheckpoint(Callback):
    def __init__(self, save_dir, save_always=False):
        super(CustomModelCheckpoint, self).__init__()
        self.save_always = save_always
        self.total_loss = np.Inf
        
        self.save_dir = save_dir
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

    def on_epoch_end(self, epoch, logs=None):
        # Save the weights only if the model performance improves
        new_loss = np.mean(logs['total_loss']) 

        if new_loss < self.total_loss or self.save_always:
            print('\nModel total_loss improved from {:.4f} to {:.4f}. Model checkpoint saved'.format(self.total_loss, new_loss))

            self.total_loss = new_loss
            self.model.save(
                # filepath=os.path.join(self.save_dir, "model_name.h5")
                filepath=self.save_dir
            )
        else:
            print('\nModel total_loss did not improve.')


class GenerateSamples(Callback):
    def __init__(self, examples_dir, save_dir, img_size=[256,256,3]):
        self.examples_dir = examples_dir
        self.save_dir = save_dir
        self.img_size = img_size

    def on_epoch_end(self, epoch, logs=None):
        epoch_dir = os.path.join(self.save_dir, str(epoch+1).zfill(3))
        if not os.path.isdir(epoch_dir):
            os.makedirs(epoch_dir)

        for example in os.listdir(self.examples_dir):
            # Load the image
            img = tf.io.read_file(os.path.join(self.examples_dir, example))
            img = tf.image.decode_jpeg(img, channels=self.img_size[2])
            img = tf.cast(img, tf.float32) / 127.5 - 1
            img = tf.reshape(img, shape=[1, *self.img_size])

            # Get a prediction and save
            prediction = self.model.generate(img)
            prediction = tf.squeeze(prediction).numpy()
            prediction = (prediction * 127.5 + 127.5).astype(np.uint8)   
            out_img = PIL.Image.fromarray(prediction)
            out_img.save(os.path.join(epoch_dir, example))