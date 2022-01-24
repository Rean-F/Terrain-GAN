import os
import datetime
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras import models
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from singan_3d.model import Discriminator, Generator

from singan_3d.tools import create_pyramid, resize_3d

class Trainer:
    def __init__(
            self,
            channels_num: int,
            num_scales: int=4,
            scales: List[float]=[0.25, 0.50, 0.75, 1.00],
            learning_rate:float=3e-4,
            n_iters: int=1000,
            noise_w: float=0.1,
            model_dir: Path=None,
            log_dir: Path=None,
            out_dir: Path=None
        ) -> None:
        """initialize SinGAN-3D trainer

        Args:
            channels_num (int): generated channels, same as embedding dims.
            num_scales (int, optional): length of scales/pyramid. Defaults to 4.
            scales (List[float], optional): scale list. Defaults to [0.25, 0.50, 0.75, 1.00].
            learning_rate (float, optional): learning rate of of training process. Defaults to 3e-4.
            n_iters (int, optional): iteration nums of training process. Defaults to 1000.
            noise_w (float, optional): init noise weight add to prev generated space. Defaults to 0.1.
            model_dir (Path, optional): model dir of singan. Defaults to None.
            log_dir (Path, optional): log dir of singan. Defaults to None.
            out_dir (Path, optional): output dir of singan. Defaults to None.
        """
        self.channels_num = channels_num
        self.num_scales = num_scales
        self.scales = scales
        self.n_iters = n_iters
        self.noise_w = noise_w
        self.model_dir = model_dir
        self.log_dir = log_dir
        self.out_dir = out_dir

        # initial_learning_rate * decay_rate ^ (step / decay_steps)
        self.lr_schedule = ExponentialDecay(
            learning_rate, decay_steps=3 * n_iters, decay_rate=0.1,
            staircase=False
        )
        self.build_model()

        self.summary_writer = tf.summary.create_file_writer(
            str(log_dir / datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        )

    def build_model(self):
        """create discriminator and generator model for every scaled space
        """
        self.discriminators: List[models.Model] = []
        self.generators: List[models.Model] = []

        for i in range(self.num_scales):
            self.discriminators.append(Discriminator(num_filters=64))
            self.generators.append(Generator(channels=self.channels_num, num_filters=64))
        return

    def save_model(self, scale: int):
        """save discriminator and generator of the specified scale level

        Args:
            scale (int): scale level index
        """
        scale_dir = self.model_dir / str(scale)
        if not scale_dir.is_dir():
            os.mkdir(scale_dir)
        G_file = scale_dir / "G" / "G"
        D_file = scale_dir / "D" / "D"
        self.generators[scale].save_weights(str(G_file), save_format="tf")
        self.discriminators[scale].save_weights(str(D_file), save_format="tf")
        np.save(self.model_dir / "noise_amps.npy", self.noise_amps)

    def load_weights(self, scale: int):
        """initialize model weights with weights of the prev scale model

        Args:
            scale (int): scale level index
        """
        prev_scale_dir = self.model_dir / str(scale - 1)
        prev_G_file = prev_scale_dir / "G" / "G"
        prev_D_file = prev_scale_dir / "D" / "D"
        self.generators[scale].load_weights(prev_G_file)
        self.discriminators[scale].load_weights(prev_D_file)

    def train(
            self,
            data: tf.Tensor,
        ):
        reals = create_pyramid(data, self.num_scales, self.scales)
        # noise weight of z_nosie added to the prev generated space
        self.noise_amps = []

        # training on each scale
        for scale in range(self.num_scales):
            print(f"training scale {scale} ...")
            # initialize model weights when scale index > 0
            if scale > 0:
                self.load_weights(scale)
            real = reals[scale]
            discriminator = self.discriminators[scale]
            generator = self.generators[scale]
            d_opt = optimizers.Adam(learning_rate=self.lr_schedule, beta_1=0.5, beta_2=0.999)
            g_opt = optimizers.Adam(learning_rate=self.lr_schedule, beta_1=0.5, beta_2=0.999)
            prev_rec = self.generate_from_coarsest_rec(reals, scale)
            # rmse gives an indication of the amount of details that needed to be added at the scale
            rmse = tf.sqrt(tf.reduce_mean(tf.square(real - prev_rec)))
            noise_amp = 1.0 if scale == 0 else self.noise_w * rmse.numpy()
            # start training on the specified scale
            for step in range(self.n_iters):
                metrics = self.train_step(
                    reals, scale, step,
                    prev_rec, noise_amp,
                    discriminator, generator,
                    d_opt, g_opt
                )
                self.write_summary(scale, step, metrics)
                if step % 500 == 0:
                    prev_rand = self.generate_from_coarsest_rand(reals, scale)
                    z = noise_amp * tf.random.uniform(real.shape, dtype=tf.float32)
                    fake_rand = generator(prev_rand, z)
                    yield scale, step, real, fake_rand
            self.noise_amps.append(noise_amp)
            self.save_model(scale)
            print(f"scale {scale} trained.")
        return

    def train_step(
            self,
            reals: List[tf.Tensor], scale: int, step: int,
            prev_rec: tf.Tensor, noise_amp: float,
            discriminator: Model, generator: Model,
            d_opt: Optimizer, g_opt: Optimizer
        ):
        """train step during specified scale level

        Args:
            reals (List[tf.Tensor]): pyramid
            scale (int): scale level
            step (int): step
            prev_rec (tf.Tensor): previous fake space with reconstruct z vector
            noise_amp (float): noise_w * rmse(real, fake_rec)
            discriminator (Model): discriminator
            generator (Model): generator
            d_opt (Optimizer): discriminator adam optimizer
            g_opt (Optimizer): generator adam optimizer
        """
        real = reals[scale]
        for i in range(6):
            # previous random generated space
            prev_rand = self.generate_from_coarsest_rand(reals, scale)
            # noise added to generator
            z_rand = noise_amp * tf.random.normal(real.shape, dtype=tf.float32)
            # reconstruct vector, aimed to generate the totally same space during the specified scale level
            z_rec = tf.random.normal(real.shape, dtype=tf.float32) if scale == 0 else tf.zeros_like(real, dtype=tf.float32)
            if i < 3:
                with tf.GradientTape() as tape:
                    fake_rand = generator(prev_rand, z_rand)
                    # WGAN-GP, loss of the discriminator
                    dis_loss = self.discriminator_wgan_loss(discriminator, real, fake_rand)
                dis_gradients = tape.gradient(dis_loss, discriminator.trainable_variables)
                d_opt.apply_gradients(zip(dis_gradients, discriminator.trainable_variables))
            else:
                with tf.GradientTape() as tape:
                    fake_rand = generator(prev_rand, z_rand)
                    fake_rec = generator(prev_rec, z_rec)
                    # WGAN-GP, loss of the generator
                    gen_wgan_loss = self.generator_wgan_loss(discriminator, fake_rand)
                    
                    # generator reconstruct loss: mse
                    gen_rec_loss = self.generator_rec_loss(real, fake_rec)
                    # larger weight of reconstruct loss means more realistic and less creative
                    gen_loss = gen_wgan_loss + 10.0 * gen_rec_loss

                    # generator reconstruct loss: cosine similarity
                    # gen_rec_loss = self.generator_rec_cosine_loss(real, fake_rec)
                    # gen_loss = gen_wgan_loss + 1.0 * gen_rec_loss
                gen_gradients = tape.gradient(gen_loss, generator.trainable_variables)
                g_opt.apply_gradients(zip(gen_gradients, generator.trainable_variables))
        metrics = [
            dis_loss, gen_loss, gen_wgan_loss, gen_rec_loss
        ]
        return metrics
    
    def generate_from_coarsest_rec(self, reals: List[tf.Tensor], scale: int) -> tf.Tensor:
        """generate reconstruct space with z-vector: [z*, 0, 0, ..., 0]

        Args:
            reals (List[tf.Tensor]): pyramid
            scale (int): stop scale level

        Returns:
            tf.Tensor: reconstruct z-vector generated fake space
        """
        fake = tf.zeros_like(reals[0])
        for i in range(scale):
            if i == 0:
                z_rec = tf.random.normal(reals[i].shape, dtype=tf.float32)
            else:
                z_rec = tf.zeros_like(reals[i], dtype=tf.float32)
            fake = self.generators[i](fake, z_rec)
            fake = resize_3d(fake, new_shape=reals[i+1].shape)
        return fake

    def generate_from_coarsest_rand(self, reals: List[tf.Tensor], scale: int) -> tf.Tensor:
        """generate random fake space

        Args:
            reals (List[tf.Tensor]): pyramid
            scale (int): stop scale level

        Returns:
            tf.Tensor: random generated fake space
        """
        fake = tf.zeros_like(reals[0])
        for i in range(scale):
            z_rand = tf.random.normal(reals[i].shape, dtype=tf.float32)
            z_rand = self.noise_amps[i] * z_rand
            fake = self.generators[i](fake, z_rand)
            fake = resize_3d(fake, new_shape=reals[i+1].shape)
        return fake

    def discriminator_wgan_loss(self, discriminator: Model, real: tf.Tensor, fake: tf.Tensor):
        # wasserstein loss
        dis_loss = tf.reduce_mean(discriminator(fake)) - tf.reduce_mean(discriminator(real))
        # gradient penalty
        alpha = tf.random.uniform(real.shape, minval=0.0, maxval=1.0, dtype=tf.float32)
        interploates = alpha * real + (1 - alpha) * fake
        with tf.GradientTape() as tape:
            tape.watch(interploates)
            dis_interploates = discriminator(interploates)
        gradients = tape.gradient(dis_interploates, [interploates])[0]
        
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[4]))
        gradient_penalty = tf.reduce_mean(tf.square(slopes - 1.0))
        # wasserstein distance + 0.1 * gradient penalty
        dis_loss = dis_loss + 0.1 * gradient_penalty
        return dis_loss

    def generator_wgan_loss(self, discriminator: Model, fake_rand: tf.Tensor):
        return -tf.reduce_mean(discriminator(fake_rand))

    def generator_rec_loss(self, real, fake_rec):
        return tf.reduce_mean(tf.square(real - fake_rec))
    
    def generator_rec_cosine_loss(self, real, fake_rec):
        return tf.reduce_mean(keras.losses.cosine_similarity(real, fake_rec, axis=-1))

    def write_summary(self, scale: int, step: int, metrics: List):
        dis_loss, gen_loss, gen_wgan_loss, gen_rec_loss = metrics
        with self.summary_writer.as_default():
            tf.summary.scalar(f"{scale}_dis_loss", dis_loss, step=step)
            tf.summary.scalar(f"{scale}_gen_loss", gen_loss, step=step)
            tf.summary.scalar(f"{scale}_gen_wgan_loss", gen_wgan_loss, step=step)
            tf.summary.scalar(f"{scale}_gen_rec_loss", gen_rec_loss, step=step)
        return