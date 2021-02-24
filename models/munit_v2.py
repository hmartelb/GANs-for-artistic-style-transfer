import argparse
import os

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers, losses


def get_compiled_model(args, verbose=False):
    img_shape = [args.img_w, args.img_h, args.img_ch]

    model = MUNIT(
        encoder_content_A=ContentEncoder(img_shape, nc_base=args.base_channels, num_layers=args.n_sample, num_residual=args.n_res),
        encoder_content_B=ContentEncoder(img_shape, nc_base=args.base_channels, num_layers=args.n_sample, num_residual=args.n_res),
        encoder_style_A=StyleEncoder(img_shape, style_dim=args.style_dim, nc_base=args.base_channels, num_layers=args.n_sample+2),
        encoder_style_B=StyleEncoder(img_shape, style_dim=args.style_dim, nc_base=args.base_channels, num_layers=args.n_sample+2),
        decoder_A=Decoder(img_shape, nc_base=args.base_channels, num_layers_content=args.n_sample, style_dim=args.style_dim, num_residual=args.n_res),
        decoder_B=Decoder(img_shape, nc_base=args.base_channels, num_layers_content=args.n_sample, style_dim=args.style_dim, num_residual=args.n_res),
        disc_A=Discriminator(img_shape, nc_base=args.base_channels, n_scale=args.n_scale, n_dis=args.n_dis),
        disc_B=Discriminator(img_shape, nc_base=args.base_channels, n_scale=args.n_scale, n_dis=args.n_dis),

        gan_w=args.gan_w,
        recon_x_w=args.recon_x_w,
        recon_s_w=args.recon_s_w,
        recon_c_w=args.recon_c_w,
        recon_x_cyc_w=args.recon_x_cyc_w,
        style_dim=args.style_dim,
        gan_type=args.gan_type,
    )

    model.compile(
        # Generator
        encoder_content_A_optimizer=tf.keras.optimizers.Adam(args.gen_lr, beta_1=0.5),
        encoder_content_B_optimizer=tf.keras.optimizers.Adam(args.gen_lr, beta_1=0.5),
        encoder_style_A_optimizer=tf.keras.optimizers.Adam(args.gen_lr, beta_1=0.5),
        encoder_style_B_optimizer=tf.keras.optimizers.Adam(args.gen_lr, beta_1=0.5),
        decoder_A_optimizer=tf.keras.optimizers.Adam(args.gen_lr, beta_1=0.5),
        decoder_B_optimizer=tf.keras.optimizers.Adam(args.gen_lr, beta_1=0.5),
        
        # Discriminator
        disc_A_optimizer=tf.keras.optimizers.Adam(args.disc_lr, beta_1=0.5),
        disc_B_optimizer=tf.keras.optimizers.Adam(args.disc_lr, beta_1=0.5),
        
        # Loss functions
        gen_loss_fn=generator_loss(gan_type=args.gan_type),
        disc_loss_fn=discriminator_loss(gan_type=args.gan_type),
    )

    return model

class MUNIT(keras.Model):
    def __init__(
        self,
        encoder_content_A,
        encoder_content_B,
        encoder_style_A,
        encoder_style_B,
        decoder_A,
        decoder_B,
        disc_A,
        disc_B,
        
        gan_w = 1.0,
        recon_x_w = 10.0,
        recon_s_w = 1.0,
        recon_c_w = 1.0,
        recon_x_cyc_w = 0.0,
        style_dim = 8,
        gan_type = 'lsgan',
    ):
        super(MUNIT, self).__init__()
        self.encoder_content_A = encoder_content_A
        self.encoder_content_B = encoder_content_B
        self.encoder_style_A = encoder_style_A
        self.encoder_style_B = encoder_style_B
        self.decoder_A = decoder_A
        self.decoder_B = decoder_B
        self.disc_A = disc_A
        self.disc_B = disc_B
        
        """ Weight """
        self.gan_w = gan_w
        self.recon_x_w = recon_x_w
        self.recon_s_w = recon_s_w
        self.recon_c_w = recon_c_w
        self.recon_x_cyc_w = recon_x_cyc_w
        
        self.style_dim = style_dim
        self.gan_type = gan_type
        
    def compile(
        self,
        encoder_content_A_optimizer,
        encoder_content_B_optimizer,
        encoder_style_A_optimizer,
        encoder_style_B_optimizer,
        decoder_A_optimizer,
        decoder_B_optimizer,
        
        disc_A_optimizer,
        disc_B_optimizer,
        
        gen_loss_fn,
        disc_loss_fn,
    ):
        super(MUNIT, self).compile()
        self.encoder_content_A_optimizer = encoder_content_A_optimizer
        self.encoder_content_B_optimizer = encoder_content_B_optimizer
        self.encoder_style_A_optimizer = encoder_style_A_optimizer
        self.encoder_style_B_optimizer = encoder_style_B_optimizer
        self.decoder_A_optimizer = decoder_A_optimizer
        self.decoder_B_optimizer = decoder_B_optimizer
        
        self.disc_A_optimizer = disc_A_optimizer
        self.disc_B_optimizer = disc_B_optimizer
        
        self.gen_loss_fn = gen_loss_fn
        self.disc_loss_fn = disc_loss_fn
    

    def generate(self, image):
        content = self.encoder_content_A(image, training=False)
        # style = self.encoder_style_A(image, training=False)
        style = np.random.normal(loc=0.0, scale=1.0, size=[content.shape[0], 1, 1, self.style_dim])
        # style = tf.random.normal(shape=(content.shape[0], 1, 1, self.style_dim))
        x_ab = self.decoder_B([content, style], training=False)
        return x_ab


    def generate_guide(self, image, reference_image):
        content = self.encoder_content_A(image, training=False)
        style = self.encoder_style_B(reference_image, training=False)
        x_ab = self.decoder_B([content, style], training=False)
        return x_ab


    def load(
        self, 
        filepath
    ):
        self.encoder_content_A = tf.saved_model.load(os.path.join(filepath, "encoder_content_A"))
        self.encoder_content_B = tf.saved_model.load(os.path.join(filepath, "encoder_content_B"))
        self.encoder_style_A = tf.saved_model.load(os.path.join(filepath, "encoder_style_A"))
        self.encoder_style_B = tf.saved_model.load(os.path.join(filepath, "encoder_style_B"))
        self.decoder_A = tf.saved_model.load(os.path.join(filepath, "decoder_A"))
        self.decoder_B = tf.saved_model.load(os.path.join(filepath, "decoder_B"))
        self.disc_A = tf.saved_model.load(os.path.join(filepath, "disc_A"))
        self.disc_B = tf.saved_model.load(os.path.join(filepath, "disc_B"))

    def save(
        self,
        filepath
    ):
        tf.saved_model.save(self.encoder_content_A, os.path.join(filepath, "encoder_content_A"))
        tf.saved_model.save(self.encoder_content_B, os.path.join(filepath, "encoder_content_B"))
        tf.saved_model.save(self.encoder_style_A, os.path.join(filepath, "encoder_style_A"))
        tf.saved_model.save(self.encoder_style_B, os.path.join(filepath, "encoder_style_B"))
        tf.saved_model.save(self.decoder_A, os.path.join(filepath, "decoder_A"))
        tf.saved_model.save(self.decoder_B, os.path.join(filepath, "decoder_B"))
        tf.saved_model.save(self.disc_A, os.path.join(filepath, "disc_A"))
        tf.saved_model.save(self.disc_B, os.path.join(filepath, "disc_B"))


    def set_lr(self, new_lr_gen, new_lr_disc=None):
        if(new_lr_disc is None):
            new_lr_disc = new_lr_gen
        
        # Set generator optimizer leraning rate
        K.set_value(self.encoder_content_A_optimizer.lr, K.get_value(new_lr_gen))
        K.set_value(self.encoder_content_B_optimizer.lr, K.get_value(new_lr_gen))
        K.set_value(self.encoder_style_A_optimizer.lr, K.get_value(new_lr_gen))
        K.set_value(self.encoder_style_B_optimizer.lr, K.get_value(new_lr_gen))
        K.set_value(self.decoder_A_optimizer.lr, K.get_value(new_lr_gen))
        K.set_value(self.decoder_B_optimizer.lr, K.get_value(new_lr_gen))

        # Set discriminator optimizer learning rate
        K.set_value(self.disc_A_optimizer.lr, K.get_value(new_lr_disc))
        K.set_value(self.disc_B_optimizer.lr, K.get_value(new_lr_disc))


    def Encoder_A(self, x):
        content_A = self.encoder_content_A(x)
        style_A = self.encoder_style_A(x)
        return content_A, style_A

    def Encoder_B(self, x):
        content_B = self.encoder_content_B(x)
        style_B = self.encoder_style_B(x)
        return content_B, style_B
        
    def discriminate_real(self, x_A, x_B):
        real_A_logit = self.disc_A(x_A)
        real_B_logit = self.disc_B(x_B)
        return real_A_logit, real_B_logit

    def discriminate_fake(self, x_ba, x_ab):
        fake_A_logit = self.disc_A(x_ba)
        fake_B_logit = self.disc_B(x_ab)
        return fake_A_logit, fake_B_logit
    
    def L1_loss(self, x, y):
        return tf.reduce_mean(tf.abs(x-y))
    
    # @tf.function
    def train_step(self, batch_data):
        real_a, real_b = batch_data
        
        batch_size = real_a.shape[0]
        # style_a = tf.random.normal(shape=(batch_size, 1, 1, self.style_dim))
        # style_b = tf.random.normal(shape=(batch_size, 1, 1, self.style_dim))
        style_a = np.random.normal(loc=0.0, scale=1.0, size=[int(batch_size), 1, 1, self.style_dim])
        style_b = np.random.normal(loc=0.0, scale=1.0, size=[int(batch_size), 1, 1, self.style_dim])
        
        with tf.GradientTape(persistent=True) as tape:
            # Encode
            content_a, style_a_prime = self.Encoder_A(real_a)
            content_b, style_b_prime = self.Encoder_B(real_b)

            # Decode (within domain)
            x_aa = self.decoder_A([content_a, style_a_prime])
            x_bb = self.decoder_B([content_b, style_b_prime])

            # Decode (cross domain)
            x_ba = self.decoder_A([content_b, style_a])
            x_ab = self.decoder_B([content_a, style_b])

            # Encode again
            content_b_, style_a_ = self.Encoder_A(x_ba)
            content_a_, style_b_ = self.Encoder_B(x_ab)

            # Decode again (if needed)
            if self.recon_x_cyc_w > 0 :
                x_aba = self.decoder_A([content_a_, style_a_prime])
                x_bab = self.decoder_B([content_b_, style_b_prime])

                cyc_recon_A = self.L1_loss(x_aba, real_a)
                cyc_recon_B = self.L1_loss(x_bab, real_b)

            else :
                cyc_recon_A = 0.0
                cyc_recon_B = 0.0

            real_A_logit, real_B_logit = self.discriminate_real(real_a, real_b)
            fake_A_logit, fake_B_logit = self.discriminate_fake(x_ba, x_ab)
            
            # Calculate the losses
            G_ad_loss_a = self.gen_loss_fn(fake_A_logit)
            G_ad_loss_b = self.gen_loss_fn(fake_B_logit)

            D_ad_loss_a = self.disc_loss_fn(real_A_logit, fake_A_logit)
            D_ad_loss_b = self.disc_loss_fn(real_B_logit, fake_B_logit)
            
            recon_A = self.L1_loss(x_aa, real_a) # reconstruction
            recon_B = self.L1_loss(x_bb, real_b) # reconstruction

            # The style reconstruction loss encourages diverse outputs given different style codes
            recon_style_A = self.L1_loss(style_a_, style_a)
            recon_style_B = self.L1_loss(style_b_, style_b)

            # The content reconstruction loss encourages the translated image to preserve semantic content of the input image
            recon_content_A = self.L1_loss(content_a_, content_a)
            recon_content_B = self.L1_loss(content_b_, content_b)

            # The final losses are calculated as a weighted sum
            Generator_A_loss = self.gan_w * G_ad_loss_a + \
                               self.recon_x_w * recon_A + \
                               self.recon_s_w * recon_style_A + \
                               self.recon_c_w * recon_content_A + \
                               self.recon_x_cyc_w * cyc_recon_A

            Generator_B_loss = self.gan_w * G_ad_loss_b + \
                               self.recon_x_w * recon_B + \
                               self.recon_s_w * recon_style_B + \
                               self.recon_c_w * recon_content_B + \
                               self.recon_x_cyc_w * cyc_recon_B

            Discriminator_A_loss = self.gan_w * D_ad_loss_a
            Discriminator_B_loss = self.gan_w * D_ad_loss_b
            
            # Generator_loss = Generator_A_loss + Generator_B_loss + self.regularization_loss('encoder') + self.regularization_loss('decoder')
            # Discriminator_loss = Discriminator_A_loss + Discriminator_B_loss + self.regularization_loss('discriminator')

        # Calculate the gradients for generators and discriminators (A and B)
        encoder_content_A_gradients = tape.gradient(Generator_A_loss, self.encoder_content_A.trainable_variables)
        encoder_content_B_gradients = tape.gradient(Generator_B_loss, self.encoder_content_B.trainable_variables)
        encoder_style_A_gradients = tape.gradient(Generator_A_loss, self.encoder_style_A.trainable_variables)
        encoder_style_B_gradients = tape.gradient(Generator_B_loss, self.encoder_style_B.trainable_variables)
        decoder_A_gradients = tape.gradient(Generator_A_loss, self.decoder_A.trainable_variables)
        decoder_B_gradients = tape.gradient(Generator_B_loss, self.decoder_B.trainable_variables)
        
        discriminator_A_gradients = tape.gradient(Discriminator_A_loss, self.disc_A.trainable_variables)
        discriminator_B_gradients = tape.gradient(Discriminator_B_loss, self.disc_B.trainable_variables)

        # Apply the gradients to the optimizer        
        self.encoder_content_A_optimizer.apply_gradients(zip(encoder_content_A_gradients, self.encoder_content_A.trainable_variables))
        self.encoder_content_B_optimizer.apply_gradients(zip(encoder_content_B_gradients, self.encoder_content_B.trainable_variables))
        self.encoder_style_A_optimizer.apply_gradients(zip(encoder_style_A_gradients, self.encoder_style_A.trainable_variables))
        self.encoder_style_B_optimizer.apply_gradients(zip(encoder_style_B_gradients, self.encoder_style_B.trainable_variables))
        self.decoder_A_optimizer.apply_gradients(zip(decoder_A_gradients, self.decoder_A.trainable_variables))
        self.decoder_B_optimizer.apply_gradients(zip(decoder_B_gradients, self.decoder_B.trainable_variables))
        
        self.disc_A_optimizer.apply_gradients(zip(discriminator_A_gradients, self.disc_A.trainable_variables))
        self.disc_B_optimizer.apply_gradients(zip(discriminator_B_gradients, self.disc_B.trainable_variables))
        
        total_loss = Generator_A_loss + Generator_B_loss + Discriminator_A_loss + Discriminator_B_loss

        return {
            "total_loss": total_loss,
            "gen_A_loss": Generator_A_loss,
            "gen_B_loss": Generator_B_loss,
            "disc_A_loss": Discriminator_A_loss,
            "disc_B_loss": Discriminator_B_loss,
        }


def ContentEncoder(img_size=[256,256,3], nc_base=64, num_layers=2, num_residual=4):
    inp = layers.Input(shape=img_size)
    x = inp
    n_channels = nc_base

    x = layers.Conv2D(
        filters=n_channels,
        kernel_size=7,
        strides=1,
        padding="same",
        kernel_initializer=tf.keras.initializers.VarianceScaling(),
        kernel_regularizer=tf.keras.regularizers.l2(0.0001),
        use_bias=True
    )(x)
    x = tfa.layers.InstanceNormalization(center=True, scale=True)(x)
    x = layers.ReLU()(x)

    for i in range(num_layers):
        x = layers.Conv2D(
            filters=n_channels*2, 
            kernel_size=4,
            strides=2,
            padding="same",
            kernel_initializer=tf.keras.initializers.VarianceScaling(),
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            use_bias=True
        )(x)
        x = tfa.layers.InstanceNormalization(center=True, scale=True)(x)
        x = layers.ReLU()(x)

        n_channels *= 2
    
    for i in range(num_residual):
        x = ResidualBlock(n_channels)(x)
    
    return keras.Model(inp, x)

def StyleEncoder(img_size=[256,256,3], style_dim=8, nc_base=64, num_layers=4):
    inp = layers.Input(shape=img_size)
    x = inp
    n_channels = nc_base

    x = layers.Conv2D(
        filters=n_channels,
        kernel_size=7,
        strides=1,
        padding="same",
        kernel_initializer=tf.keras.initializers.VarianceScaling(),
        kernel_regularizer=tf.keras.regularizers.l2(0.0001),
        use_bias=True
    )(x)
    x = layers.ReLU()(x)

    for i in range(num_layers):
        x = layers.Conv2D(
            filters=n_channels*2,
            kernel_size=4, 
            strides=2,
            padding="same",
            kernel_initializer=tf.keras.initializers.VarianceScaling(),
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            use_bias=True
        )(x)
        x = layers.ReLU()(x)

        # Last 2 layers keep channels constant
        if(i < num_layers-2):
            n_channels *= 2

    x = AdaptiveAvgPooling()(x) 
    x = layers.Conv2D(
        filters=style_dim,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer=tf.keras.initializers.VarianceScaling(),
        kernel_regularizer=tf.keras.regularizers.l2(0.0001),
        use_bias=True
    )(x)
    return keras.Model(inp, x)

def Decoder(img_size=[256,256,3], nc_base=64, num_layers_content=2, style_dim=8, num_residual=4):
    n_channels = nc_base * (2**num_layers_content)
    content_shape = [img_size[0] // (2**num_layers_content), img_size[1] // (2**num_layers_content), n_channels]
    style_shape = [1, 1, style_dim]

    inp_c = layers.Input(shape=content_shape)
    inp_s = layers.Input(shape=style_shape)

    x = inp_c
    mu, var = MLP(nc_base=nc_base, num_layers_content=num_layers_content, num_residual=num_residual)(inp_s)

    for i in range(num_residual):
        idx = 2*i
        x = AdaptiveResblock(filters=n_channels)(x, gamma1=mu[idx], beta1=var[idx], gamma2=mu[idx+1], beta2=var[idx+1])
    
    for i in range(num_layers_content):
        x = layers.UpSampling2D(size=(2, 2))(x)
        x = layers.Conv2D(
            filters=n_channels//2,
            kernel_size=5,
            strides=1,
            padding="same",
            kernel_initializer=tf.keras.initializers.VarianceScaling(),
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            use_bias=True
        )(x)
        x = layers.LayerNormalization(center=True, scale=True)(x)
        x = layers.ReLU()(x)

        n_channels /= 2

    x = layers.Conv2D(
        filters=img_size[2],
        kernel_size=7,
        strides=1,
        padding="same",
        kernel_initializer=tf.keras.initializers.VarianceScaling(),
        kernel_regularizer=tf.keras.regularizers.l2(0.0001),
        use_bias=True,
        activation="tanh"
    )(x)
    return keras.Model([inp_c, inp_s], x)


def Discriminator(img_size=[256,256,3], nc_base=64, n_scale=3, n_dis=4):
    D_logit = []
    inp = layers.Input(shape=img_size)
    x_init = inp

    for scale in range(n_scale):
        n_channels = nc_base
        x = x_init
        for i in range(n_dis):
            x = layers.Conv2D(
                filters=n_channels,
                kernel_size=4,
                strides=2,
                padding="same",
                kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
                kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                use_bias=True
            )(x)
            x = layers.LeakyReLU(0.2)(x)
            if(i > 0):
                n_channels *= 2

            x = layers.Conv2D(
                filters=1,
                kernel_size=1,
                strides=1,
                padding="same",
                kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
                kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                use_bias=True
            )(x)
            D_logit.append(x)

        # Downsample x_init
        x_init = layers.AveragePooling2D(pool_size=(3,3), strides=2, padding="same")(x_init)
    
    return keras.Model(inp, D_logit)

def MLP(nc_base=64, num_layers_content=2, mlp_layers=2, num_residual=4):
    def f(x):
        n_channels = nc_base * (2**num_layers_content)

        x = layers.Flatten()(x)
        for i in range(mlp_layers):
            x = layers.Dense(n_channels, activation='relu')(x)
        
        mu_list, var_list = [], []
        for i in range(2*num_residual):
            mu = layers.Dense(n_channels)(x)
            var = layers.Dense(n_channels)(x)

            mu = layers.Reshape(target_shape=[1, 1, n_channels])(mu)
            var = layers.Reshape(target_shape=[1, 1, n_channels])(var)

            mu_list.append(mu)
            var_list.append(var)

        return mu_list, var_list
    return f

def AdaptiveAvgPooling():
    def f(x):
        x = layers.Lambda(lambda x: tf.reduce_mean(x, axis=[1,2], keepdims=True))(x)
        return x
    return f

def AdaptiveResblock(filters, kernel_size=3, strides=1, use_norm=False, use_bias=True, smoothing=True, name=None):
    def f(x, gamma1, beta1, gamma2, beta2):
        inp = x
        for i in range(2):
            x = layers.Conv2D(
                filters=filters, 
                kernel_size=kernel_size, 
                strides=strides, 
                use_bias=use_bias,
                padding="same"
            )(x)
            gamma, beta = (gamma1, beta1) if(i==0) else (gamma2, beta2)
            x = AdaptiveInstanceNormalization()(x, gamma, beta)
            if(i==0):
                x = layers.ReLU()(x)    
        return layers.Add()([x, inp])
    return f

def AdaptiveInstanceNormalization(epsilon=1e-5):
    def f(x, gamma, beta):
        mean, var = layers.Lambda(lambda x: tf.nn.moments(x, axes=[1,2], keepdims=True))(x)
        std = layers.Lambda(lambda x: tf.sqrt(x+epsilon))(var)
        return gamma * ((x-mean) / std) + beta
    return f


def ResidualBlock(filters, kernel_size=3, strides=1, use_norm=True, use_bias=True, n_conv=2):
    def f(x):
        inp = x
        for _ in range(n_conv-1):
            x = Conv2DBlock(
                filters=filters, 
                kernel_size=kernel_size, 
                strides=strides, 
                use_norm=use_norm, 
                use_bias=use_bias
            )(x)
        # Last convolution does not use activation
        x = Conv2DBlock(
            filters=filters, 
            kernel_size=kernel_size, 
            strides=strides, 
            use_norm=use_norm, 
            use_bias=use_bias, 
            use_activation=False # <--
        )(x)
        return layers.Add()([x, inp])
    return f


def Conv2DBlock(filters, kernel_size=4, strides=2, use_norm=False, use_bias=False, use_activation=True, w_l2=1e-4, sn=False):
    def f(x):
        if sn:
            x = tfa.layers.SpectralNormalization(
                layers.Conv2D(
                    filters=filters,  
                    kernel_size=kernel_size,
                    strides=strides, 
                    kernel_initializer='he_normal',
                    kernel_regularizer=keras.regularizers.l2(w_l2),
                    use_bias=(not use_norm or use_bias), 
                    padding="same"
                )
            )(x)
        else:
            x = layers.Conv2D(
                filters=filters,  
                kernel_size=kernel_size,
                strides=strides, 
                kernel_initializer='he_normal',
                kernel_regularizer=keras.regularizers.l2(w_l2),
                use_bias=(not use_norm or use_bias), 
                padding="same"
            )(x)
        if use_norm:
            x = tfa.layers.InstanceNormalization(epsilon=1e-5)(x)
        if use_activation:
            x = layers.ReLU()(x)
        return x
    return f

def discriminator_loss(gan_type='lsgan'):
    def f(real, generated):
        n_scale = len(real)
        loss = []
        real_loss, generated_loss = 0, 0
        for i in range(n_scale) :
            if(gan_type == 'lsgan'):
                real_loss = tf.reduce_mean(tf.square(tf.ones_like(real[i]) - real[i]))
                generated_loss = tf.reduce_mean(tf.square(generated[i]))
            if(gan_type == 'gan'):
                real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real[i]), logits=real[i]))
                generated_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(generated[i]), logits=generated[i]))
            loss.append(real_loss + generated_loss)
        return sum(loss)
    return f

def generator_loss(gan_type='lsgan'):
    def f(generated):
        n_scale = len(generated)
        loss = []
        generated_loss = 0
        for i in range(n_scale) :
            if(gan_type == 'lsgan'):
                generated_loss = tf.reduce_mean(tf.square(tf.ones_like(generated[i]) - generated[i]))
            if(gan_type == 'gan'):
                generated_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(generated[i]), logits=generated[i]))
            loss.append(generated_loss)
        return sum(loss)
    return f

# def discriminator_loss(use_lsgan=True):
#     def f(real, generated):
#         if use_lsgan:
#             real_loss = tf.keras.losses.MSE(tf.ones_like(real), real)
#             generated_loss = tf.keras.losses.MSE(tf.zeros_like(generated), generated)
#         else:
#             
#         total_disc_loss = real_loss + generated_loss
#         return total_disc_loss * 0.5
#     return f


# def generator_loss(use_lsgan=True):
#     def f(generated):
#         if use_lsgan:
#             loss = tf.keras.losses.MSE(tf.ones_like(generated), generated)
#         else:
#             loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(generated), generated)
#         return loss
#     return f



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    
    # Data parameters
    ap.add_argument('--dataset', required=False, default="data")
    ap.add_argument('--img_w', type=int, default=256, help='The size of image width')
    ap.add_argument('--img_h', type=int, default=256, help='The size of image hegiht')
    ap.add_argument('--img_ch', type=int, default=3, help='The size of image channel')
    ap.add_argument('--augment', type=bool, default=False, help='Use data augmentation or not (True / False)')

    # Hyperparameters
    ap.add_argument('--gen_lr', required=False, default=1e-4)
    ap.add_argument('--disc_lr', required=False, default=1e-4)

    # Model parameters
    ap.add_argument('--gan_type', type=str, default='lsgan', help='GAN loss type [gan / lsgan]')
    ap.add_argument('--gan_w', type=float, default=1.0, help='Weight of adversarial loss')
    ap.add_argument('--recon_x_w', type=float, default=10.0, help='Weight of image reconstruction loss')
    ap.add_argument('--recon_s_w', type=float, default=1.0, help='Weight of style reconstruction loss')
    ap.add_argument('--recon_c_w', type=float, default=1.0, help='Weight of content reconstruction loss')
    ap.add_argument('--recon_x_cyc_w', type=float, default=0.3, help='Weight of explicit style augmented cycle consistency loss')

    ap.add_argument('--style_dim', type=int, default=8, help='Length of style code')

    args = ap.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

    #############
    # Unit tests
    #############
    img_shape = [args.img_w, args.img_h, args.img_ch]

    # style_encoder = StyleEncoder(img_shape)
    # style_encoder.summary()

    # content_encoder = ContentEncoder(img_shape)
    # content_encoder.summary()

    # decoder = Decoder(img_shape)
    # decoder.summary()

    # discriminator = Discriminator(img_shape)
    # discriminator.summary()

    model = get_compiled_model(args)

    ones = tf.ones(shape=[1, *img_shape])

    # c,s = content_encoder(ones), style_encoder(ones)
    # pred = decoder([c,s])

    # print(ones.shape, c.shape, s.shape, pred.shape)

    generated = model.generate(ones)
    print("Generated", generated.shape)

    content = model.encoder_content_A(ones)
    style = model.encoder_style_A(ones)
    print("[Content, Style]", content.shape, style.shape)

    decoded = model.decoder_A([content, style])
    print("Decoded", decoded.shape)

    step_dict = model.train_step([ones, ones])
    print(step_dict)
