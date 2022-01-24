from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from singan_3d.model import Generator
from singan_3d.tools import create_pyramid, resize_3d


class Inferencer:
    def __init__(
            self,
            channels_num: int,
            num_scales: int,
            scales: List[float],
            model_dir: Path,
            out_dir: Path,
        ) -> None:
        """initialize SinGAN-3D inferencer

        Args:
            channels_num (int): generated channels, same as embedding dims.
            num_scales (int): length of scales/pyramid
            scales (List[float]): scale list
            model_dir (Path): model dir of singan
            out_dir (Path): output dir of singan
        """
        self.generators = []
        self.noise_amps = []
        self.channels_num = channels_num
        self.num_scales = num_scales
        self.scales = scales
        self.out_dir = out_dir

        self.load_model(model_dir)

    def load_model(self, model_dir: Path):
        """load noise amplitudes and generators

        Args:
            model_dir (Path): model dir of singan
        """
        self.noise_amps = np.load(model_dir / "noise_amps.npy")
        for scale in range(self.num_scales):
            generator = Generator(channels=self.channels_num, num_filters=64)
            generator.load_weights(model_dir / str(scale) / "G" / "G")
            self.generators.append(generator)
        return

    def random_generate(self, real: tf.Tensor, inject_scale: int=None) -> tf.Tensor:
        """random generate space. While inject_scale is not none,
        the fake generated space less than the inject_scale is replaced by real.

        Args:
            real (tf.Tensor): real embeded space
            inject_scale (int, optional): inject scale. Defaults to None.

        Returns:
            tf.Tensor: fake space, 5D tensor, [batch_size, x, y, z, c]
        """
        reals = create_pyramid(real, self.num_scales, self.scales)
        fake = tf.zeros_like(reals[0], dtype=tf.float32)
        for scale in range(self.num_scales):
            if inject_scale is not None and scale <= inject_scale:
                fake = reals[scale]
            else:
                fake = resize_3d(fake, new_shape=reals[scale].shape)
                z = tf.random.uniform(reals[scale].shape, dtype=tf.float32)
                z = self.noise_amps[scale] * z
                fake = self.generators[scale](fake, z)
        return fake

    def rand_size_generate(self, dims: Tuple[int]) -> tf.Tensor:
        """generate random size space

        Args:
            dims (Tuple[int]): new space shape

        Returns:
            tf.Tensor: fake space, 5D tensor, [batch_size, new_x, new_y, new_z, c]
        """
        pyramid_dims = []
        for scale in self.scales:
            xdim = int(dims[0] * scale)
            ydim = int(dims[1] * scale)
            zdim = int(dims[2] * scale)
            pyramid_dims.append([1, xdim, ydim, zdim, self.channels_num])
        fake = tf.zeros(pyramid_dims[0], dtype=tf.float32)
        for scale in range(self.num_scales):
            fake = resize_3d(fake, new_shape=pyramid_dims[scale])
            z = tf.random.uniform(fake.shape, dtype=tf.float32)
            z = self.noise_amps[scale] * z
            fake = self.generators[scale](fake, z)
        return fake