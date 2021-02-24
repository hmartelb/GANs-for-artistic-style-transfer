import argparse
import os

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers, losses


def get_compiled_model(args, verbose=False):
    img_shape = [args.img_w, args.img_h, args.img_ch]

    model = MUNIT(
        encoder_content_A=Encoder_content_MUNIT(img_shape),
        encoder_content_B=Encoder_content_MUNIT(img_shape),
        encoder_style_A=Encoder_style_MUNIT(img_shape),
        encoder_style_B=Encoder_style_MUNIT(img_shape),
        decoder_A=Decoder_MUNIT([img_shape[0], img_shape[1], 256]),
        decoder_B=Decoder_MUNIT([img_shape[0], img_shape[1], 256]),
        disc_A=Discriminator(img_shape),
        disc_B=Discriminator(img_shape),

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
        gen_loss_fn=generator_loss(use_lsgan=(args.gan_type == 'lsgan')),
        disc_loss_fn=discriminator_loss(use_lsgan=(args.gan_type == 'lsgan')),
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
        style = self.encoder_style_A(image, training=False)
        content = self.encoder_content_A(image, training=False)
        x_ab, _, _ = self.decoder_B([style, content], training=False)
        return x_ab


    def generate_guide(self, image, reference_image):
        style = self.encoder_style_B(reference_image, training=False)
        content = self.encoder_content_A(image, training=False)
        x_ab, _, _ = self.decoder_B([style, content], training=False)
        return x_ab


    def load(
        self, 
        filepath
    ):
        self.encoder_content_A.load_weights(filepath.replace('model_name', "encoder_content_A"), by_name=True),
        self.encoder_content_B.load_weights(filepath.replace('model_name', "encoder_content_B"), by_name=True),
        self.encoder_style_A.load_weights(filepath.replace('model_name', "encoder_style_A"), by_name=True),
        self.encoder_style_B.load_weights(filepath.replace('model_name', "encoder_style_B"), by_name=True),
        self.decoder_A.load_weights(filepath.replace('model_name', "decoder_A"), by_name=True),
        self.decoder_B.load_weights(filepath.replace('model_name', "decoder_B"), by_name=True),
        self.disc_A.load_weights(filepath.replace('model_name', "disc_A"), by_name=True),
        self.disc_B.load_weights(filepath.replace('model_name', "disc_B"), by_name=True),

    def save(
        self,
        filepath
    ):
        self.encoder_content_A.save(filepath.replace('model_name', "encoder_content_A")),
        self.encoder_content_B.save(filepath.replace('model_name', "encoder_content_B")),
        self.encoder_style_A.save(filepath.replace('model_name', "encoder_style_A")),
        self.encoder_style_B.save(filepath.replace('model_name', "encoder_style_B")),
        self.decoder_A.save(filepath.replace('model_name', "decoder_A")),
        self.decoder_B.save(filepath.replace('model_name', "decoder_B")),
        self.disc_A.save(filepath.replace('model_name', "disc_A")),
        self.disc_B.save(filepath.replace('model_name', "disc_B")),

    def Encoder_A(self, x_A):
        style_A = self.encoder_style_A(x_A)
        content_A = self.encoder_content_A(x_A)
        return content_A, style_A

    def Encoder_B(self, x_B):
        style_B = self.encoder_style_B(x_B)
        content_B = self.encoder_content_B(x_B)
        return content_B, style_B
        
    def discriminate_real(self, x_A, x_B):
        real_A_logit = self.disc_A(x_A)
        real_B_logit = self.disc_B(x_B)
        return real_A_logit, real_B_logit

    def discriminate_fake(self, x_ba, x_ab):
        fake_A_logit = self.disc_A(x_ba)
        fake_B_logit = self.disc_B(x_ab)
        return fake_A_logit, fake_B_logit
    
    def L1_loss(self,x,y):
        return tf.reduce_mean(tf.abs(x-y))
    
    @tf.function
    def train_step(self, batch_data):
        real_monet, real_photo = batch_data
        
        batch_size = tf.shape(real_monet)[0]
        
        style_a = tf.random.normal(shape=(batch_size, self.style_dim))
        style_b = tf.random.normal(shape=(batch_size, self.style_dim))
        
        with tf.GradientTape(persistent=True) as tape:
            # Encode
            content_a, style_a_prime = self.Encoder_A(real_photo)
            content_b, style_b_prime = self.Encoder_B(real_monet)

            # Decode (within domain)
            x_aa, _, _ = self.decoder_A([style_a_prime, content_a])
            x_bb, _, _ = self.decoder_B([style_b_prime, content_b])

            # Decode (cross domain)
            x_ba, _, _ = self.decoder_A([style_a, content_b])
            x_ab, _, _ = self.decoder_B([style_b, content_a])

            # Encode again
            content_b_, style_a_ = self.Encoder_A(x_ba)
            content_a_, style_b_ = self.Encoder_B(x_ab)

            # Decode again (if needed)
            if self.recon_x_cyc_w > 0 :
                x_aba, _, _ = self.decoder_A([style_a_prime, content_a_])
                x_bab, _, _ = self.decoder_B([style_b_prime, content_b_])

                cyc_recon_A = self.L1_loss(x_aba, real_photo)
                cyc_recon_B = self.L1_loss(x_bab, real_monet)

            else :
                cyc_recon_A = 0.0
                cyc_recon_B = 0.0

            real_A_logit, real_B_logit = self.discriminate_real(real_photo, real_monet)
            fake_A_logit, fake_B_logit = self.discriminate_fake(x_ba, x_ab)
            
            # Calculate the losses
            G_ad_loss_a = self.gen_loss_fn(fake_A_logit)
            G_ad_loss_b = self.gen_loss_fn(fake_B_logit)

            D_ad_loss_a = self.disc_loss_fn(real_A_logit, fake_A_logit)
            D_ad_loss_b = self.disc_loss_fn(real_B_logit, fake_B_logit)
            
            recon_A = self.L1_loss(x_aa, real_photo) # reconstruction
            recon_B = self.L1_loss(x_bb, real_monet) # reconstruction

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


def Encoder_style_MUNIT(img_size=[256,256,3], n_dim_adain=256, n_dim_style=8, nc_base=64, n_dowscale_style=4, w_l2=1e-4):
    # Style encoder architecture 
    inp = layers.Input(shape=img_size)
#     x = ReflectPadding2D(inp, 3)
    x = inp
    x = layers.Conv2D(
        filters=64, 
        # kernel_size=7, 
        # kernel_size=4,
        kernel_size=3,
        kernel_initializer='he_normal', 
        kernel_regularizer=keras.regularizers.l2(w_l2),
        use_bias=True, 
        padding="same"
    )(x)
    x = layers.ReLU()(x)  
    
    dim = 1
    for i in range(n_dowscale_style):
        dim = 4 if dim >= 4 else dim*2
        x = conv_block(x, dim*nc_base)
        
    x = layers.GlobalAveragePooling2D()(x)    
    style_code = layers.Dense(n_dim_style, kernel_regularizer=keras.regularizers.l2(w_l2))(x) # Style code
    return keras.Model(inp, style_code)


def Encoder_content_MUNIT(img_size=[256,256,3], n_downscale_content=2, nc_base=64, w_l2=1e-4, n_resblocks=3):
    # Content encoder architecture 
    def res_block_content(input_tensor, f):
        x = input_tensor
#         x = ReflectPadding2D(x)
        x = layers.Conv2D(
            filters=f, 
            # kernel_size=4, # 3
            kernel_size=3,
            kernel_initializer='he_normal', 
            kernel_regularizer=keras.regularizers.l2(w_l2),
            use_bias=False, 
            padding="same"
        )(x)
        x = tfa.layers.InstanceNormalization(epsilon=1e-5)(x)
        x = layers.ReLU()(x)
#         x = ReflectPadding2D(x)
        x = layers.Conv2D(
            filters=f, 
            # kernel_size=4, # 3
            kernel_size=3,
            kernel_initializer='he_normal', 
            kernel_regularizer=keras.regularizers.l2(w_l2),
            use_bias=False, 
            padding="same"
        )(x)
        x = tfa.layers.InstanceNormalization(epsilon=1e-5)(x)
        x = layers.Add()([x, input_tensor])
        return x      
    
    inp = layers.Input(shape=img_size)
#     x = ReflectPadding2D(inp, 3)
    x = inp
    x = layers.Conv2D(
        filters=64, 
        # kernel_size=7, 
        # kernel_size=4,
        kernel_size=3,
        kernel_initializer='he_normal', 
        kernel_regularizer=keras.regularizers.l2(w_l2),
        use_bias=False, 
        padding="same"
    )(x)
    x = tfa.layers.InstanceNormalization()(x) #
    x = layers.ReLU()(x)
    
    dim = 1
    ds = 2**n_downscale_content
    for i in range(n_downscale_content):
        dim = 4 if dim >= 4 else dim*2
        x = conv_block(x, dim*nc_base, use_norm=True)
    for i in range(n_resblocks):
        x = res_block_content(x, dim*nc_base)
    content_code = x # Content code
    return keras.Model(inp, content_code)


def MLP_MUNIT(n_dim_style=8, n_dim_adain=256, n_blk=3, n_adain=6, w_l2=1e-4):
    # MLP for AdaIN parameters
    inp_style_code = layers.Input(shape=(n_dim_style,))
    adain_params = layers.Dense(n_dim_adain, kernel_regularizer=keras.regularizers.l2(w_l2), activation='relu')(inp_style_code)
    for i in range(n_blk - 2):
        adain_params = layers.Dense(n_dim_adain, kernel_regularizer=keras.regularizers.l2(w_l2), activation='relu')(adain_params)
    adain_params = layers.Dense(2*n_adain*n_dim_adain, kernel_regularizer=keras.regularizers.l2(w_l2))(adain_params) # No output activation 
    return keras.Model(inp_style_code, [adain_params])
  

def Decoder_MUNIT(img_size=[256//(2**2), 256//(2**2), 256], nc_in=256, n_dim_adain=256, n_resblocks=3, n_downscale_content=2, n_dim_style=8, use_groupnorm=False, w_l2=1e-4):
    def op_adain(inp):
        x = inp[0]
        adain_bias = inp[1]
        adain_weight = inp[2]

        mean, var = tf.nn.moments(x, [1,2], keepdims=True)
        adain_bias = K.reshape(adain_bias, (-1, 1, 1, n_dim_adain))
        adain_weight = K.reshape(adain_weight, (-1, 1, 1, n_dim_adain))      

        out = tf.nn.batch_normalization(x, mean, var, adain_bias, adain_weight, variance_epsilon=1e-7)
        return out
      
    def AdaptiveInstanceNorm2d(inp, adain_params, idx_adain):
        assert inp.shape[-1] == n_dim_adain
        x = inp

        idx_head = idx_adain*2*n_dim_adain
        adain_weight = layers.Lambda(lambda x: x[:, idx_head:idx_head+n_dim_adain])(adain_params)
        adain_bias = layers.Lambda(lambda x: x[:, idx_head+n_dim_adain:idx_head+2*n_dim_adain])(adain_params)

        out = layers.Lambda(op_adain)([x, adain_bias, adain_weight])
        return out
      
    def res_block_adain(inp, f, adain_params, idx_adain, w_l2=1e-4):
        x = inp
#         x = ReflectPadding2D(x)
        x = layers.Conv2D(
            filters=f, 
            # kernel_size=4, # 3
            kernel_size=3,
            kernel_initializer='he_normal', 
            kernel_regularizer=keras.regularizers.l2(w_l2),
            bias_regularizer=keras.regularizers.l2(w_l2),
            use_bias=False, 
            padding="same"
        )(x)

        x = layers.Lambda(lambda params: AdaptiveInstanceNorm2d(params[0], params[1], idx_adain))([x, adain_params])     
        x = layers.ReLU()(x)
#         x = ReflectPadding2D(x)
        x = layers.Conv2D(
            filters=f, 
            # kernel_size=4, # 3
            kernel_size=3,
            kernel_initializer='he_normal', 
            kernel_regularizer=keras.regularizers.l2(w_l2), 
            bias_regularizer=keras.regularizers.l2(w_l2),
            use_bias=False, 
            padding="same"
        )(x)
        x = layers.Lambda(lambda params: AdaptiveInstanceNorm2d(params[0], params[1], idx_adain+1))([x, adain_params])    
        
        x = layers.Add()([x, inp])
        return x  
    
    inp_style = layers.Input((n_dim_style,))
    style_code = inp_style
    mlp = MLP_MUNIT(n_dim_style)
    adain_params = mlp(style_code)
    
    inp_content = layers.Input(shape=[img_size[0] // 4, img_size[1] // 4, img_size[2]])
    content_code = inp_content
    x = inp_content
    
    for i in range(n_resblocks):
        x = res_block_adain(x, nc_in, adain_params, 2*i) 
        
    dim = 1
    for i in range(n_downscale_content):
        dim = dim if nc_in//dim <= 64 else dim*2
        # Note: original MUNIT uses layer norm instead of group norm in upscale blocks
        x = upscale_nn(x, nc_in//dim, use_norm=True)
#     x = ReflectPadding2D(x, 3)
    out = layers.Conv2D(
        filters=3, 
        # kernel_size=7, 
        # kernel_size=4,
        kernel_size=3,
        kernel_initializer='he_normal', 
        kernel_regularizer=keras.regularizers.l2(w_l2), 
        padding='same', 
        activation="tanh"
    )(x)
    return keras.Model([inp_style, inp_content], [out, style_code, content_code])


def Discriminator(img_size=[256,256,3], w_l2=1e-4, use_lsgan=False):
    inp = layers.Input(shape=img_size)
    x = conv_block_d(inp, 64, False)
    x = conv_block_d(x, 128, False)
    x = conv_block_d(x, 256, False)
#     x = ReflectPadding2D(x, 2)
    out = layers.Conv2D(
        filters=1, 
        # kernel_size=5, 
        # kernel_size=4,
        kernel_size=3,
        kernel_initializer=keras.initializers.RandomNormal(0,0.02), 
        kernel_regularizer=keras.regularizers.l2(w_l2),
        use_bias=False, 
        padding="same"
    )(x)  
    if not use_lsgan:
        x = layers.Activation('sigmoid')(x) 
    return keras.Model(inputs=[inp], outputs=out)


def Discriminator_MS(img_size=[256,256,3], w_l2=1e-4, use_lsgan=False):
    # Multi-scale discriminator architecture
    inp = layers.Input(shape=img_size)
    
    def conv2d_blocks(inp, nc_base=64, n_layers=3):
        x = inp
        dim = nc_base
        for _ in range(n_layers):
            x = conv_block_d(x, dim, False)
            dim *= 2
        x = layers.Conv2D(
            filters=1, 
            kernel_size=1, 
            kernel_initializer=keras.initializers.RandomNormal(0,0.02),
            kernel_regularizer=keras.regularizers.l2(w_l2),
            use_bias=True, 
            padding="same"
        )(x)
        if not use_lsgan:
            x = layers.Activation('sigmoid')(x)
        return x
    
    x0 = conv2d_blocks(inp)    
    ds1 = layers.AveragePooling2D(pool_size=(3, 3), strides=2)(inp)
    x1 = conv2d_blocks(ds1)
    ds2 = layers.AveragePooling2D(pool_size=(3, 3), strides=2)(ds1)
    x2 = conv2d_blocks(ds2)
    return keras.Model(inputs=[inp], outputs=[x0, x1, x2])


def conv_block(input_tensor, f, k=4, strides=2, use_norm=False, w_l2=1e-4):
    x = input_tensor
#     x = ReflectPadding2D(x)
    x = layers.Conv2D(
        filters=f, 
        # kernel_size=4, 
        kernel_size=3,
        strides=strides, 
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(w_l2),
        use_bias=(not use_norm), 
        padding="same"
    )(x)
    if use_norm:
        x = tfa.layers.InstanceNormalization(epsilon=1e-5)(x)
    x = layers.ReLU()(x)
    return x


def conv_block_d(input_tensor, f, use_norm=False, w_l2=1e-4):
    x = input_tensor
#     x = ReflectPadding2D(x, 2)
    x = layers.Conv2D(
        filters=f, 
        # kernel_size=4, 
        kernel_size=3,
        strides=2, 
        kernel_initializer=keras.initializers.RandomNormal(0, 0.02),
        kernel_regularizer=keras.regularizers.l2(w_l2),
        use_bias=(not use_norm), 
        padding="same"
    )(x)
    if use_norm:
        x = tfa.layers.InstanceNormalization(epsilon=1e-5)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    return x


def upscale_nn(inp, f, use_norm=False, w_l2=1e-4, use_groupnorm=False):
    x = inp
    x = layers.UpSampling2D()(x)
#     x = ReflectPadding2D(x, 2)
    x = layers.Conv2D(
        filters=f, 
        # kernel_size=5, 
        # kernel_size=4,
        kernel_size=3,
        kernel_initializer='he_normal', 
        kernel_regularizer=keras.regularizers.l2(w_l2), 
        use_bias=(not use_norm), 
        padding='same'
    )(x)
    if use_norm:
        if use_groupnorm:
            x = tfa.layers.GroupNormalization(groups=32)(x)
        else:
            x = tfa.layers.GroupNormalization(groups=f)(x) # group=f equivalant to layer norm
    x = layers.ReLU()(x)
    return x

  
def ReflectPadding2D(x, pad=1):
    x = layers.Lambda(lambda x: tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT'))(x)
    return x


def discriminator_loss(use_lsgan=True):
    def f(real, generated):
        if use_lsgan:
            real_loss = tf.keras.losses.MSE(tf.ones_like(real), real)
            generated_loss = tf.keras.losses.MSE(tf.zeros_like(generated), generated)
        else:
            real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(real), real)
            generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.zeros_like(generated), generated)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss * 0.5
    return f


def generator_loss(use_lsgan=True):
    def f(generated):
        if use_lsgan:
            loss = tf.keras.losses.MSE(tf.ones_like(generated), generated)
        else:
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(generated), generated)
        return loss
    return f

def calc_cycle_loss(real_image, cycled_image, LAMBDA):
    return LAMBDA * tf.reduce_mean(tf.abs(real_image - cycled_image))


def identity_loss(real_image, same_image, LAMBDA):
    return LAMBDA * 0.5 * tf.reduce_mean(tf.abs(real_image - same_image))


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

    if False:
        encoder_style = Encoder_style_MUNIT(img_shape)
        encoder_style.summary()

        encoder_content = Encoder_content_MUNIT(img_shape)
        encoder_content.summary()

        decoder = Decoder_MUNIT([img_shape[0], img_shape[1], 256])
        decoder.summary()

        disc = Discriminator(img_shape)
        disc.summary()

        disc_ms = Discriminator_MS(img_shape)
        disc_ms.summary()

    model = get_compiled_model(args)

    ones = tf.ones(shape=[1, *img_shape])
    generated = model.generate(ones)
    print("Generated", generated.shape)

    content = model.encoder_content_A(ones)
    style = model.encoder_style_A(ones)
    print("[Content, Style]", content.shape, style.shape)

    decoded, _, _ = model.decoder_A([style, content])
    print("Decoded", decoded.shape)

    step_dict = model.train_step([ones, ones])
    print(step_dict)