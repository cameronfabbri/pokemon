"""

"""
#
#
import os
import random

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

import network
import utils.tf_ops as tfo
import utils.data_ops as do

from pokemon_data import PokemonData


def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image.shape[-1] == 4:
        trans_mask = image[:, :, 3] == 0
        image[trans_mask] = [255, 255, 255, 255]
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image

def main():

    batch_size = 2
    learning_rate = 0.0001
    style_dim = 16

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    seed_value = 3
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)

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

    generator = network.Generator()
    discriminator = network.Discriminator(c_dim=4)
    mapping_network = network.MappingNetwork(c_dim=4, style_dim=style_dim)
    encoder = network.Encoder(c_dim=4, style_dim=style_dim)

    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.0, beta_2=0.99)
    #generator_optimizer = tfa.optimizers.MovingAverage(generator_optimizer)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.0, beta_2=0.99)
    mapping_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6, beta_1=0.0, beta_2=0.99)

    os.makedirs('model', exist_ok=True)
    checkpoint = tf.train.Checkpoint(
            encoder=encoder,
            generator=generator,
            discriminator=discriminator,
            mapping_network=mapping_network,
            mapping_optimizer=mapping_optimizer,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer)
    manager = tf.train.CheckpointManager(checkpoint, directory='model', max_to_keep=1)

    def GT():
        return tf.GradientTape()

    @tf.function
    def trainingStep(
            batch_images_x,
            batch_label,
            batch_label1,
            batch_label2):

        """
        batch_labels_y: Tensor of shape (batch_size, 1)
        """
        with GT() as g_tape, GT() as d_tape, GT() as m_tape:

            # These are for the adversarial loss
            random_style_z = tf.random.normal((batch_size, style_dim), dtype=tf.float32)
            random_style_s = tf.gather(mapping_network(random_style_z), batch_label)

            # Generate a batch of images given the random style
            batch_images_g = generator(batch_images_x, random_style_s)

            # GAN loss
            d_fake = discriminator(batch_images_g)
            d_real = discriminator(batch_images_x)

            errG = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=tf.ones_like(d_fake), logits=d_fake))

            errD_real = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=d_real, labels=tf.ones_like(d_real))
            errD_fake = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=d_fake, labels=tf.zeros_like(d_real))
            errD = tf.reduce_mean(errD_real + errD_fake)

            real_loss = tf.reduce_mean(d_real)
            real_grads = tf.gradients(real_loss, batch_images_x)[0]

            r1_penalty = tf.reduce_mean(tf.square(real_grads))
            errD += r1_penalty
            #errG = tf.reduce_mean(d_fake)
            #errD = tf.reduce_mean(tf.nn.relu(1 + d_real) + tf.nn.relu(1 - d_fake))

            # Encode the style given the generated images
            encoded_style_s = tf.gather(encoder(batch_images_g), batch_label)
            style_recon_loss = tf.reduce_mean(tf.abs(random_style_s - encoded_style_s))

            # Style diversification loss
            random_style_z1 = tf.random.normal((batch_size, style_dim), dtype=tf.float32)
            random_style_z2 = tf.random.normal((batch_size, style_dim), dtype=tf.float32)
            random_style_s1 = tf.gather(mapping_network(random_style_z1), batch_label1)
            random_style_s2 = tf.gather(mapping_network(random_style_z2), batch_label2)
            batch_images_g1 = generator(batch_images_x, random_style_s1)
            batch_images_g2 = generator(batch_images_x, random_style_s2)

            # Want to maximize this
            style_div_loss = -tf.reduce_mean(tf.abs(batch_images_g1-batch_images_g2))

            # Cycle Consistency loss
            batch_images_cyc = generator(batch_images_g, encoded_style_s)
            cyc_loss = tf.reduce_mean(tf.abs(batch_images_g-batch_images_cyc))

            # Gradient penalty
            '''
            epsilon = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0., maxval=1.)
            x_hat = batch_images_x + epsilon * (batch_images_g - batch_images_x)
            d_hat = discriminator(x_hat)
            grad_d_hat = tf.gradients(d_hat, [x_hat])[0]
            slopes = tf.sqrt(1e-8 + tf.reduce_sum(tf.square(grad_d_hat), axis=[1, 2, 3]))
            gradient_penalty = 10*tf.reduce_mean((slopes - 1.) ** 2)
            errD += gradient_penalty
            '''

            total_errG = errG + style_recon_loss + style_div_loss + cyc_loss

        gen_tv = generator.trainable_variables + encoder.trainable_variables

        gradients_g = g_tape.gradient(total_errG, gen_tv)
        generator_optimizer.apply_gradients(zip(gradients_g, gen_tv))

        gradients_d = d_tape.gradient(errD, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_d, discriminator.trainable_variables))

        gradients_m = m_tape.gradient(errG, mapping_network.trainable_variables)
        mapping_optimizer.apply_gradients(zip(gradients_m, mapping_network.trainable_variables))

        return {
            'errG': errG,
            'errD': errD,
            'style_recon_loss': style_recon_loss,
            'style_div_loss': style_div_loss,
            'cyc_loss': cyc_loss
        }

    def get_batch(
            data_dict,
            batch_labels):

        batch_images_x = np.empty((batch_size, 64, 64, 3), dtype=np.float32)

        for i, label in enumerate(batch_labels):
            path = random.choice(data_dict[label])
            image = load_image(path)
            image = cv2.resize(image, (64, 64)).astype(np.float32)
            image = (image / 127.5) - 1.0
            batch_images_x[i, ...] = image

        return tf.convert_to_tensor(batch_images_x)


    gens = [1,2,4,5]

    test_path = random.choice(test_data_dict[5])

    for step in range(1000000):

        # These labels are used to get different image classes
        batch_labels_y = [random.choice(gens) for i in range(batch_size)]
        batch_images_x = get_batch(
                train_data_dict,
                batch_labels_y)

        # This label is used in training
        batch_label = tf.convert_to_tensor(np.array([random.choice(gens)]))
        batch_label1 = tf.convert_to_tensor(np.array([random.choice(gens)]))
        batch_label2 = tf.convert_to_tensor(np.array([random.choice(gens)]))

        step_info = trainingStep(
                batch_images_x,
                batch_label,
                batch_label1,
                batch_label2)

        errG = step_info['errG'].numpy()
        errD = step_info['errD'].numpy()
        recon_loss = step_info['style_recon_loss'].numpy()
        div_loss = step_info['style_div_loss'].numpy()
        cyc_loss = step_info['cyc_loss'].numpy()

        statement = ' | step: ' + str(step)
        statement += ' | errG: %.2f' % errG
        statement += ' | errD: %.2f' % errD
        statement += ' | recon_loss: %.3f' % recon_loss
        statement += ' | div_loss: %.3f' % div_loss
        statement += ' | cyc_loss: %.3f' % cyc_loss

        print(statement)

        if step % 10 == 0:

            manager.save()

            image = load_image(test_path)
            image = cv2.resize(image, (64, 64)).astype(np.float32)
            image = (image / 127.5) - 1.0
            original_image = ((image+1.0)*127.5).astype(np.uint8)
            gen_images = [original_image]

            batch_images_x = tf.expand_dims(image, 0)

            for gen in gens:

                batch_label = tf.convert_to_tensor(np.array([gen]))
                style_z = tf.random.normal((batch_size, style_dim), dtype=tf.float32)
                style_s = tf.gather(mapping_network(style_z), batch_label)
                image_g = generator(batch_images_x, style_s)[0].numpy()
                image_g = ((image_g+1.0)*127.5).astype(np.uint8)
                gen_images.append(image_g)

            canvas = cv2.hconcat(gen_images)
            cv2.imwrite(os.path.join('model', 'canvas_'+str(step)+'.png'), canvas)


if __name__ == '__main__':
    main()
