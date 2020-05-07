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
import utils.losses as losses

from pokemon_data import PokemonData


def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image.shape[-1] == 4:
        trans_mask = image[:, :, 3] == 0
        image[trans_mask] = [255, 255, 255, 255]
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def get_data():

    data_dir = os.path.join('data','pokemon','done')
    pd = PokemonData(data_dir)

    gen1_paths = pd.get_paths_from_gen(1)
    gen2_paths = pd.get_paths_from_gen(2)
    gen4_paths = pd.get_paths_from_gen(4)
    gen5_paths = pd.get_paths_from_gen(5)

    random.shuffle(gen1_paths)
    random.shuffle(gen2_paths)
    random.shuffle(gen4_paths)
    random.shuffle(gen5_paths)

    gen1_train_paths = np.asarray(gen1_paths[:int(0.95*len(gen1_paths))])
    gen2_train_paths = np.asarray(gen2_paths[:int(0.95*len(gen1_paths))])
    gen4_train_paths = np.asarray(gen4_paths[:int(0.95*len(gen1_paths))])
    gen5_train_paths = np.asarray(gen5_paths[:int(0.95*len(gen1_paths))])

    gen1_test_paths = np.asarray(gen1_paths[int(0.95*len(gen1_paths)):])
    gen2_test_paths = np.asarray(gen2_paths[int(0.95*len(gen1_paths)):])
    gen4_test_paths = np.asarray(gen4_paths[int(0.95*len(gen1_paths)):])
    gen5_test_paths = np.asarray(gen5_paths[int(0.95*len(gen1_paths)):])

    train_data_dict = {
            1: gen1_train_paths,
            2: gen2_train_paths,
            4: gen4_train_paths,
            5: gen5_train_paths
    }

    test_data_dict = {
            1: gen1_test_paths,
            2: gen2_test_paths,
            4: gen4_test_paths,
            5: gen5_test_paths
    }

    return train_data_dict, test_data_dict


def main():

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    seed_value = 3
    tf.random.set_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)

    gens = [1, 2, 4, 5]

    batch_size = 2
    num_iters = 100000
    style_dim = 16
    c_dim = 4

    # Network learning rates
    lr_g = 0.0001
    lr_d = 0.0001
    lr_e = 0.0001
    lr_f = 0.0001

    # Loss function weightings
    r1_gamma = 1.0
    lambda_sty = 1.0
    lambda_cyc = 1.0
    lambda_ds_start = 2.0

    use_ema = False

    train_data_dict, test_data_dict = get_data()

    # Define networks
    network_g = network.Generator()
    network_d = network.Discriminator(c_dim=c_dim)
    network_f = network.MappingNetwork(c_dim=c_dim, style_dim=style_dim)
    network_e = network.Encoder(c_dim=c_dim, style_dim=style_dim)

    # Trainable variables for each network
    tv_g = network_g.trainable_variables
    tv_d = network_d.trainable_variables
    tv_e = network_e.trainable_variables
    tv_f = network_f.trainable_variables

    # Optimizers
    g_opt = tf.keras.optimizers.Adam(learning_rate=lr_g, beta_1=0.0, beta_2=0.99)
    d_opt = tf.keras.optimizers.Adam(learning_rate=lr_d, beta_1=0.0, beta_2=0.99)
    e_opt = tf.keras.optimizers.Adam(learning_rate=lr_e, beta_1=0.0, beta_2=0.99)
    f_opt = tf.keras.optimizers.Adam(learning_rate=lr_f, beta_1=0.0, beta_2=0.99)

    # Exponential moving average
    if use_ema:
        g_opt = tfa.optimizers.MovingAverage(g_opt)
        e_opt = tfa.optimizers.MovingAverage(e_opt)
        f_opt = tfa.optimizers.MovingAverage(f_opt)

    os.makedirs('model', exist_ok=True)
    checkpoint = tf.train.Checkpoint(
            network_e=network_e,
            network_g=network_g,
            network_d=network_d,
            network_f=network_f,
            g_opt=g_opt,
            d_opt=d_opt,
            e_opt=e_opt,
            f_opt=f_opt)
    manager = tf.train.CheckpointManager(
            checkpoint, directory='model', max_to_keep=1)

    def GT():
        return tf.GradientTape()

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
            image = (image / 127.5) - 1.0
            batch_images_x[i, ...] = image

        return tf.convert_to_tensor(batch_images_x)

    test_path = random.choice(test_data_dict[5])

    @tf.function
    def single_step(
            batch_images_x,
            batch_labels_org,
            batch_labels_trg):

        # Image that will go through generator
        image_x_real = tf.expand_dims(batch_images_x[batch_n], 0)

        label_org = tf.squeeze(batch_labels_org[batch_n], axis=[0,1])
        label_trg = tf.squeeze(batch_labels_trg[batch_n], axis=[0,1])

        style_z = tf.random.normal((1, style_dim), dtype=tf.float32)
        style_z1 = tf.random.normal((1, style_dim), dtype=tf.float32)
        style_z2 = tf.random.normal((1, style_dim), dtype=tf.float32)

        style_s = tf.gather(network_f(style_z), label_trg)
        style_s1 = tf.gather(network_f(style_z1), label_trg)
        style_s2 = tf.gather(network_f(style_z2), label_trg)

        image_x_fake = network_g(image_x_real, style_s)
        image_x_fake1 = network_g(image_x_real, style_s1)
        image_x_fake2 = network_g(image_x_real, style_s2)

        # Style vector generated by the encoder network on real data
        style_e_real = tf.gather(network_e(image_x_real), label_org) # (1, 16)

        # Style vector generated by the encoder network on fake data
        style_e_fake = tf.gather(network_e(image_x_fake), label_trg)

        # Cycle-consistency. Image generated from fake image and real encoded style
        image_x_cyc = network_g(image_x_fake, style_e_real)

        # Output from network_d on real and fake data
        d_real = network_d(image_x_real)
        d_fake = network_d(image_x_fake)

        # ~~ Losses ~~ #

        g_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(d_fake), logits=d_fake))

        errD_real = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_real, labels=tf.ones_like(d_real))
        errD_fake = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_fake, labels=tf.zeros_like(d_fake))
        d_loss = tf.reduce_mean(errD_real + errD_fake)

        # R1 reg loss for D
        gradients = tf.gradients(d_real, [image_x_real])[0]
        r1_penalty = tf.reduce_mean((r1_gamma / 2) * tf.square(gradients))

        d_loss += r1_penalty

        # Style reconstruction loss
        rec_loss = tf.reduce_mean(tf.abs(style_s - style_e_fake))

        # Diversity loss
        div_loss = ds_w * tf.reduce_mean(tf.abs(image_x_fake1 - image_x_fake2))

        # Cycle loss
        cyc_loss = tf.reduce_mean(tf.abs(image_x_cyc - image_x_real))

        return g_loss, d_loss, rec_loss, div_loss, cyc_loss

    test_labels_org = [tf.convert_to_tensor((tf.reshape(x, [1,1]))) for x in gens]
    test_labels_trg = [tf.convert_to_tensor((tf.reshape(x, [1,1]))) for x in gens]

    for step in range(1, num_iters):

        ds_w = lambda_ds_start * (num_iters - step) / (num_iters - 1)

        # These labels are used to get different image classes
        batch_labels_org = [random.choice(gens) for i in range(batch_size)]
        batch_labels_trg = [random.choice(gens) for i in range(batch_size)]
        batch_images_x = get_batch(
                train_data_dict,
                batch_labels_org)

        batch_labels_org = [tf.convert_to_tensor((tf.reshape(x, [1,1]))) for x in batch_labels_org]
        batch_labels_trg = [tf.convert_to_tensor((tf.reshape(x, [1,1]))) for x in batch_labels_trg]

        #with GT() as g_tape, GT() as d_tape, GT() as e_tape, GT() as f_tape:
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape, tf.GradientTape() as e_tape, tf.GradientTape() as f_tape:

            errG = []
            errD = []
            rec_loss = []
            div_loss = []
            cyc_loss = []

            for batch_n in range(batch_size):

                res = single_step(
                        batch_images_x,
                        batch_labels_org,
                        batch_labels_trg)

                errG.append(res[0])
                errD.append(res[1])
                rec_loss.append(res[2])
                div_loss.append(res[3])
                cyc_loss.append(res[4])

            g_loss = tf.reduce_mean(errG)
            d_loss = tf.reduce_mean(errD)
            cyc_loss = tf.reduce_mean(cyc_loss)
            rec_loss = tf.reduce_mean(rec_loss)
            div_loss = tf.reduce_mean(div_loss)

        gradients_g = g_tape.gradient(g_loss, tv_g)
        gradients_d = d_tape.gradient(d_loss, tv_d)
        gradients_e = e_tape.gradient(g_loss, tv_e)
        gradients_f = f_tape.gradient(g_loss, tv_f)

        print(gradients_g)
        print(tf.reduce_mean(gradients_g))
        print(tf.norm(gradients_g),'\n')

        g_opt.apply_gradients(zip(gradients_g, tv_g))
        d_opt.apply_gradients(zip(gradients_d, tv_d))
        e_opt.apply_gradients(zip(gradients_e, tv_e))
        f_opt.apply_gradients(zip(gradients_f, tv_f))

        statement = ' | step: ' + str(step)
        statement += ' | errG: %.2f' % g_loss
        statement += ' | errD: %.2f' % d_loss
        statement += ' | rec_loss: %.3f' % rec_loss
        statement += ' | div_loss: %.3f' % div_loss
        statement += ' | cyc_loss: %.3f' % cyc_loss

        #print(statement)

        if step % 2 == 0:

            manager.save()

            image = load_image(test_path)
            image = cv2.resize(image, (64, 64)).astype(np.float32)
            image = (image / 127.5) - 1.0
            original_image = ((image+1.0)*127.5).astype(np.uint8)
            gen_images = [original_image]

            image_x_real = tf.expand_dims(image, 0)

            for n in range(len(gens)):

                label_org = tf.squeeze(test_labels_org[n], axis=[0,1])
                label_trg = tf.squeeze(test_labels_trg[n], axis=[0,1])

                style_z = tf.random.normal((1, style_dim), dtype=tf.float32)
                style_s = tf.gather(network_f(style_z), label_trg)

                image_x_fake = network_g(image_x_real, style_s)[0].numpy()
                image_x_fake = ((image_x_fake+1.0)*127.5).astype(np.uint8)
                gen_images.append(image_x_fake)

            canvas = cv2.hconcat(gen_images)
            cv2.imwrite(os.path.join('model', 'canvas_'+str(step)+'.png'), canvas)


if __name__ == '__main__':
    main()
