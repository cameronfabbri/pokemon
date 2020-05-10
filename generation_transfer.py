"""

"""
#
#
import os
import random

import cv2
import numpy as np
import tensorflow_addons as tfa

import network
import tensorflow as tf
import utils.data_ops as do

from pokemon_data import PokemonData


def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image.shape[-1] == 4:
        trans_mask = image[:, :, 3] == 0
        image[trans_mask] = [255, 255, 255, 255]
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    image = do.crop_image(image)
    return image


def get_afhq():
    data_dir = '/home/cameron/Research/datasets/data/afhq'


def get_data(gens):

    data_dir = os.path.join('data','pokemon','done')
    pd = PokemonData(data_dir)

    train_data_dict = {}
    test_data_dict = {}

    if 1 in gens:
        gen1_paths = pd.get_paths_from_gen(1)
        random.shuffle(gen1_paths)
        gen1_train_paths = np.asarray(gen1_paths[:int(0.95*len(gen1_paths))])
        gen1_test_paths = np.asarray(gen1_paths[int(0.95*len(gen1_paths)):])
        train_data_dict[1] = gen1_train_paths
        test_data_dict[1] = gen1_test_paths

    if 2 in gens:
        gen2_paths = pd.get_paths_from_gen(2)
        random.shuffle(gen2_paths)
        gen2_train_paths = np.asarray(gen2_paths[:int(0.95*len(gen1_paths))])
        gen2_test_paths = np.asarray(gen2_paths[int(0.95*len(gen1_paths)):])
        train_data_dict[2] = gen2_train_paths
        test_data_dict[2] = gen2_test_paths

    if 4 in gens:
        gen4_paths = pd.get_paths_from_gen(4)
        random.shuffle(gen4_paths)
        gen4_train_paths = np.asarray(gen4_paths[:int(0.95*len(gen1_paths))])
        gen4_test_paths = np.asarray(gen4_paths[int(0.95*len(gen1_paths)):])
        train_data_dict[4] = gen4_train_paths
        test_data_dict[4] = gen4_test_paths

    if 5 in gens:
        gen5_paths = pd.get_paths_from_gen(5)
        random.shuffle(gen5_paths)
        gen5_train_paths = np.asarray(gen5_paths[:int(0.95*len(gen1_paths))])
        gen5_test_paths = np.asarray(gen5_paths[int(0.95*len(gen1_paths)):])
        train_data_dict[5] = gen5_train_paths
        test_data_dict[5] = gen5_test_paths

    return train_data_dict, test_data_dict


def main():

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    seed_value = 3
    tf.random.set_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)

    dataset = 'afhq'
    dataset = 'pokemon'

    if dataset == 'pokemon':
        # Using gen1 and gen5
        gens = {
            0: 1,
            1: 5,
        }

        train_data_dict, test_data_dict = get_data(list(gens.values()))
        c_dim = len(list(gens.keys()))

    elif dataset == 'afhq':
        train_data_dict, test_data_dict = get_afhq()

    save_freq = 100
    batch_size = 2
    num_iters = 100000
    latent_dim = 16 # 16
    style_dim = 64 # 64

    # Network learning rates
    lr_g = 1e-4
    lr_d = 1e-4
    lr_e = 1e-4
    lr_f = 1e-6

    # Loss function weightings
    r1_gamma = 1.0
    lambda_sty = 1.0
    lambda_cyc = 1.0
    lambda_ds_start = 1.0
    lambda_reg = 1.0

    # Weight for high pass filter
    w_hpf = 1.

    use_ema = True

    # Define networks
    network_g = network.Generator()
    network_d = network.Discriminator(c_dim=c_dim)
    network_f = network.MappingNetwork(c_dim=c_dim, style_dim=style_dim)
    network_e = network.Encoder(c_dim=c_dim, style_dim=style_dim)

    # Testing networks that use the parameters from EMA
    #test_network_g = network.Generator()
    #test_network_e = network.Encoder(c_dim=c_dim, style_dim=style_dim)
    #test_network_f = network.MappingNetwork(c_dim=c_dim, style_dim=style_dim)

    # Optimizers
    #g_opt = tf.keras.optimizers.Adam(learning_rate=lr_g, beta_1=0.0, beta_2=0.99)
    #d_opt = tf.keras.optimizers.Adam(learning_rate=lr_d, beta_1=0.0, beta_2=0.99)
    #e_opt = tf.keras.optimizers.Adam(learning_rate=lr_e, beta_1=0.0, beta_2=0.99)
    #f_opt = tf.keras.optimizers.Adam(learning_rate=lr_f, beta_1=0.0, beta_2=0.99)

    g_opt = tfa.optimizers.AdamW(learning_rate=lr_g, beta_1=0.0, beta_2=0.99, weight_decay=1e-4)
    d_opt = tfa.optimizers.AdamW(learning_rate=lr_d, beta_1=0.0, beta_2=0.99, weight_decay=1e-4)
    e_opt = tfa.optimizers.AdamW(learning_rate=lr_e, beta_1=0.0, beta_2=0.99, weight_decay=1e-4)
    f_opt = tfa.optimizers.AdamW(learning_rate=lr_f, beta_1=0.0, beta_2=0.99, weight_decay=1e-4)

    # Exponential moving average
    g_opt = tfa.optimizers.MovingAverage(g_opt)
    e_opt = tfa.optimizers.MovingAverage(e_opt)
    f_opt = tfa.optimizers.MovingAverage(f_opt)

    os.makedirs('models/', exist_ok=True)
    os.makedirs('models/network_g/', exist_ok=True)
    os.makedirs('models/network_d/', exist_ok=True)
    os.makedirs('models/network_e/', exist_ok=True)
    os.makedirs('models/network_f/', exist_ok=True)

    def get_batch(
            data_dict,
            batch_labels):

        batch_images_x = np.empty((batch_size, 64, 64, 3), dtype=np.float32)

        for i, label in enumerate(batch_labels):
            path = random.choice(data_dict[label])
            image = load_image(path)
            r = random.random()
            if r < 0.5:
                image = np.fliplr(image)
            image = cv2.resize(image, (64, 64)).astype(np.float32)
            image = do.normalize(image)
            batch_images_x[i, ...] = image

        return tf.convert_to_tensor(batch_images_x)

    @tf.function
    def training_step_d_map(
            x_real,
            y_org,
            y_trg,
            z_trg):
        """
        Function to train the discriminator using fake images generated using
        the style network
        """

        with tf.GradientTape() as d_tape:

            # Real images
            d_real = network_d(x_real, y_org)

            d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_real, labels=tf.ones_like(d_real))

            # R1 reg loss for D
            gradients = tf.gradients(d_real, [x_real])[0]
            d_loss_reg = tf.reduce_mean((r1_gamma / 2) * tf.square(gradients))

            s_trg = network_f(z_trg, y_trg)

            # Fake images using mapping network
            x_fake = network_g(x_real, s_trg)

            d_fake = network_d(x_fake, y_trg)

            d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_fake, labels=tf.zeros_like(d_fake))

            d_loss = d_loss_real + d_loss_fake + (lambda_reg * d_loss_reg)

        gradients_d = d_tape.gradient(d_loss, network_d.trainable_variables)
        d_opt.apply_gradients(zip(gradients_d, network_d.trainable_variables))

        return d_loss

    @tf.function
    def training_step_d_ref(
            x_real,
            y_org,
            y_trg,
            x_ref):
        """
        Function to train the discriminator using the encoder to generate a
        style code given a reference image
        """

        with tf.GradientTape() as d_tape:

            # Real images
            d_real = network_d(x_real, y_org)
            d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_real, labels=tf.ones_like(d_real))

            # R1 reg loss for D
            gradients = tf.gradients(d_real, [x_real])[0]
            d_loss_reg = tf.reduce_mean((r1_gamma / 2) * tf.square(gradients))

            # Generate style code using encoder on reference image
            s_trg = network_e(x_ref, y_trg)

            x_fake = network_g(x_real, s_trg)

            d_fake = network_d(x_fake, y_trg)

            d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_fake, labels=tf.zeros_like(d_fake))

            d_loss = d_loss_real + d_loss_fake + (lambda_reg * d_loss_reg)

        gradients_d = d_tape.gradient(d_loss, network_d.trainable_variables)
        d_opt.apply_gradients(zip(gradients_d, network_d.trainable_variables))

        return d_loss

    @tf.function
    def training_step_g_map(
            x_real,
            y_org,
            y_trg,
            z_trg,
            z_trg2,
            lambda_ds):

        with tf.GradientTape() as g_tape, tf.GradientTape() as e_tape, tf.GradientTape() as f_tape:

            s_trg = network_f(z_trg, y_trg)

            x_fake = network_g(x_real, s_trg)

            d_fake = network_d(x_fake, y_trg)

            loss_g = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=tf.ones_like(d_fake), logits=d_fake))

            # Style reconstruction loss
            s_pred = network_e(x_fake, y_trg)
            loss_sty = tf.reduce_mean(tf.abs(s_pred - s_trg))

            # Diversity loss
            s_trg2 = network_f(z_trg2, y_trg)

            x_fake2 = network_g(x_real, s_trg2)

            loss_ds = tf.reduce_mean(tf.abs(x_fake - x_fake2))

            # cycle consistency loss
            s_org = network_e(x_real, y_org)
            x_rec = network_g(x_fake, s_org)

            loss_cyc = tf.reduce_mean(tf.abs(x_rec - x_real))

            loss = loss_g + (lambda_sty*loss_sty) - (lambda_ds*loss_ds) + (lambda_cyc*loss_cyc)

        gradients_g = g_tape.gradient(loss, network_g.trainable_variables)
        gradients_e = e_tape.gradient(loss, network_e.trainable_variables)
        gradients_f = f_tape.gradient(loss, network_f.trainable_variables)

        g_opt.apply_gradients(zip(gradients_g, network_g.trainable_variables))
        e_opt.apply_gradients(zip(gradients_e, network_e.trainable_variables))
        f_opt.apply_gradients(zip(gradients_f, network_f.trainable_variables))

        return loss_g, loss_sty, loss_ds, loss_cyc

    @tf.function
    def training_step_g_ref(
            x_real,
            y_org,
            y_trg,
            x_ref,
            x_ref2,
            lambda_ds):

        with tf.GradientTape() as g_tape, tf.GradientTape() as e_tape, tf.GradientTape() as f_tape:

            s_trg = network_e(x_ref, y_trg)

            x_fake = network_g(x_real, s_trg)

            d_fake = network_d(x_fake, y_trg)

            loss_g = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=tf.ones_like(d_fake), logits=d_fake))

            # Style reconstruction loss
            s_pred = network_e(x_fake, y_trg)
            loss_sty = tf.reduce_mean(tf.abs(s_pred - s_trg))

            # Diversity loss
            s_trg2 = network_e(x_ref2, y_trg)

            x_fake2 = network_g(x_real, s_trg2)

            loss_ds = tf.reduce_mean(tf.abs(x_fake - x_fake2))

            # cycle consistency loss
            s_org = network_e(x_real, y_org)
            x_rec = network_g(x_fake, s_org)

            loss_cyc = tf.reduce_mean(tf.abs(x_rec - x_real))

            loss = loss_g + (lambda_sty*loss_sty) - (lambda_ds*loss_ds) + (lambda_cyc*loss_cyc)

        # TODO - they don't optimize network_e here for some reason
        gradients_g = g_tape.gradient(loss, network_g.trainable_variables)
        #gradients_e = e_tape.gradient(loss, network_e.trainable_variables)

        g_opt.apply_gradients(zip(gradients_g, network_g.trainable_variables))
        #e_opt.apply_gradients(zip(gradients_e, network_e.trainable_variables))

        return loss_g, loss_sty, loss_ds, loss_cyc

    def training_step(
            x_real,
            y_org,
            x_ref,
            x_ref2,
            y_trg,
            z_trg,
            z_trg2,
            lambda_ds):
        """
        """

        # ~~ Train the discriminator using mapping network ~~ #
        d_loss1 = training_step_d_map(
                x_real=x_real,
                y_org=y_org,
                y_trg=y_trg,
                z_trg=z_trg)

        # ~~ Train the discriminator using the encoder network with reference image ~~ #
        d_loss2 = training_step_d_ref(
                x_real=x_real,
                y_org=y_org,
                y_trg=y_trg,
                x_ref=x_ref)

        # ~~ Train the generator using the mapping network ~~ #
        res = training_step_g_map(
                x_real,
                y_org,
                y_trg,
                z_trg,
                z_trg2,
                lambda_ds)
        (loss_g1, loss_sty1, loss_ds1, loss_cyc1) = res

        # ~~ Train the generator using the encoder network with reference image ~~ #
        res = training_step_g_ref(
                x_real,
                y_org,
                y_trg,
                x_ref,
                x_ref2,
                lambda_ds)
        (loss_g2, loss_sty2, loss_ds2, loss_cyc2) = res

        loss_g = (loss_g1+loss_g2)/2.
        d_loss = tf.reduce_mean((d_loss1+d_loss2)/2.)
        loss_sty = (loss_sty1 + loss_sty2)/2.
        loss_ds = (loss_ds1+loss_ds2)/2.
        loss_cyc = (loss_cyc1 + loss_cyc2)/2.

        return loss_g, d_loss, loss_sty, loss_ds, loss_cyc

    test_y_gen1 = [0]
    test_y_gen5 = [1]

    test_x_gen1 = get_batch(
            test_data_dict,
            [gens[i] for i in test_y_gen1])

    test_x_gen5 = get_batch(
            test_data_dict,
            [gens[i] for i in test_y_gen5])

    test_y_gen1 = tf.convert_to_tensor([[n,x] for n, x in enumerate(test_y_gen5)])
    test_y_gen5 = tf.convert_to_tensor([[n,x] for n, x in enumerate(test_y_gen5)])

    test_z = tf.random.normal((batch_size, latent_dim), dtype=tf.float32)
    test_x_gen1_im = np.squeeze(do.unnormalize(test_x_gen1[0].numpy()).astype(np.uint8))
    test_x_gen5_im = np.squeeze(do.unnormalize(test_x_gen5[0].numpy()).astype(np.uint8))

    for step in range(1, num_iters):

        lambda_ds = tf.convert_to_tensor(lambda_ds_start * (num_iters - step) / (num_iters - 1))

        # x_real goes with y_org
        # x_ref goes with y_trg
        # x_ref2 goes with y_trg

        y_org = [random.choice(list(gens.keys())) for i in range(batch_size)]
        y_trg = [random.choice(list(gens.keys())) for i in range(batch_size)]

        x_real = get_batch(
                train_data_dict,
                [gens[i] for i in y_org])

        x_ref = get_batch(
                train_data_dict,
                [gens[i] for i in y_trg])

        x_ref2 = get_batch(
                train_data_dict,
                [gens[i] for i in y_trg])

        z_trg = tf.random.normal((batch_size, latent_dim), dtype=tf.float32)
        z_trg2 = tf.random.normal((batch_size, latent_dim), dtype=tf.float32)

        y_org = tf.convert_to_tensor([[n,x] for n, x in enumerate(y_org)])
        y_trg = tf.convert_to_tensor([[n,x] for n, x in enumerate(y_trg)])

        res = training_step(
                x_real=x_real,
                y_org=y_org,
                x_ref=x_ref,
                x_ref2=x_ref2,
                y_trg=y_trg,
                z_trg=z_trg,
                z_trg2=z_trg2,
                lambda_ds=lambda_ds)

        loss_g = res[0]
        d_loss = res[1]
        rec_loss = res[2]
        div_loss = res[3]
        cyc_loss = res[4]

        statement = ' | step: ' + str(step)
        statement += ' | errG: %.5f' % loss_g
        statement += ' | errD: %.5f' % d_loss
        statement += ' | rec_loss: %.5f' % rec_loss
        statement += ' | div_loss: %.5f' % div_loss
        statement += ' | cyc_loss: %.5f' % cyc_loss
        print(statement)

        if step % 10 == 0:

            print('Saving out models...\n')

            # Save model weights (non averaged)
            network_g.save_weights('models/network_g/model', save_format='tf')
            network_d.save_weights('models/network_d/model', save_format='tf')
            network_e.save_weights('models/network_e/model', save_format='tf')
            network_f.save_weights('models/network_f/model', save_format='tf')

            # Average the model parameters
            g_opt.assign_average_vars(network_g.variables)
            e_opt.assign_average_vars(network_e.variables)
            f_opt.assign_average_vars(network_f.variables)

            # Test the model using the averaged weights

            # Generate a fake gen1 image given a gen5 image and a style code for gen1
            test_style_code = network_f(test_z, test_y_gen1)
            test_x_fake1 = network_g(test_x_gen5, test_style_code)

            # Generate a fake gen5 image given a gen1 image and a style code for gen5
            test_style_code = network_f(test_z, test_y_gen5)
            test_x_fake2 = network_g(test_x_gen1, test_style_code)

            # Generate a fake gen1 image given a reference gen5 image
            test_style_code = network_e(test_x_gen1, test_y_gen1)
            test_x_fake3 = network_g(test_x_gen5, test_style_code)

            # Generate a fake gen5 image given a reference gen1 image
            test_style_code = network_e(test_x_gen5, test_y_gen5)
            test_x_fake4 = network_g(test_x_gen1, test_style_code)

            test_x_fake1 = np.squeeze(do.unnormalize(test_x_fake1[0].numpy()).astype(np.uint8))
            test_x_fake2 = np.squeeze(do.unnormalize(test_x_fake2[0].numpy()).astype(np.uint8))
            test_x_fake3 = np.squeeze(do.unnormalize(test_x_fake3[0].numpy()).astype(np.uint8))
            test_x_fake4 = np.squeeze(do.unnormalize(test_x_fake4[0].numpy()).astype(np.uint8))

            # Create canvas and save image
            canvas1 = cv2.hconcat([test_x_gen1_im, test_x_gen5_im])
            canvas2 = cv2.hconcat([test_x_fake1, test_x_fake2])
            canvas3 = cv2.hconcat([test_x_fake3, test_x_fake4])
            canvas = cv2.vconcat([canvas1, canvas2, canvas3])
            cv2.imwrite(os.path.join('models', 'canvas_'+str(step)+'.png'), canvas)

            # Load back up the non-averaged weights to continue training with
            network_g.load_weights('models/network_g/model')
            network_e.load_weights('models/network_e/model')
            network_f.load_weights('models/network_f/model')




if __name__ == '__main__':
    main()
