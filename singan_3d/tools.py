from typing import List, Tuple

import tensorflow as tf
from tensorflow_graphics.math.interpolation import trilinear

def resize_3d(
        data: tf.Tensor,
        new_shape: Tuple[int, int, int]=None,
        scale_factor: float=None,
        min_size: int=0
    ) -> tf.Tensor:
    """scale 3D space using grid sample and trilinear

    Args:
        data (tf.Tensor): 3D space, 5D Tensor, [batch_size, x, y, z, c]
        new_shape (Tuple[int, int, int], optional): [new_x, new_y, new_z]. Defaults to None.
        scale_factor (float, optional): zoom level. Defaults to None.
        min_size (int, optional): x/y/z minimum value. Defaults to 0.

    Raises:
        Exception: should give new_shape or scale_factor

    Returns:
        tf.Tensor: scaled 3D space, [batch_size, new_x, new_y, new_z, c]
    """
    if new_shape is not None:
        new_xdim = new_shape[1]
        new_ydim = new_shape[2]
        new_zdim = new_shape[3]
    elif scale_factor is not None:
        new_xdim = max(int(data.shape[1] * scale_factor), min_size)
        new_ydim = max(int(data.shape[2] * scale_factor), min_size)
        new_zdim = max(int(data.shape[3] * scale_factor), min_size)
    else:
        raise Exception("please specific new_shape or scale_factor.")
    batch_size = data.shape[0]
    max_x = data.shape[1] - 1
    max_y = data.shape[2] - 1
    max_z = data.shape[3] - 1
    chs_size = data.shape[4]

    # grid sample
    x = tf.linspace(-1, 1, new_xdim)
    y = tf.linspace(-1, 1, new_ydim)
    z = tf.linspace(-1, 1, new_zdim)
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    z = tf.cast(z, tf.float32)
    x = 0.5 * (x + 1.0) * tf.cast(max_x, tf.float32)
    y = 0.5 * (y + 1.0) * tf.cast(max_y, tf.float32)
    z = 0.5 * (z + 1.0) * tf.cast(max_z, tf.float32)

    meshx, meshy, meshz = tf.meshgrid(x, y, z, indexing="ij")
    grids = tf.stack([meshx, meshy, meshz], axis=3)

    # trilinear interpolate
    new_data = trilinear.interpolate(data, tf.reshape(grids, [batch_size, -1, 3]))
    # reshape to 5D tensor
    new_data = tf.reshape(new_data, [batch_size, new_xdim, new_ydim, new_zdim, chs_size])
    return new_data

def create_pyramid(
    data: tf.Tensor,
    num_scales: int,
    scales: List[float],
    min_size: int=0
) -> Tuple[tf.Tensor]:
    """zoom on each scale

    Args:
        data (tf.Tensor): 3D space, 5D tensor
        num_scales (int): length of scales list
        scales (List[float]): scale list, eg. [0.25, 0.50, 0.75, 1.00]
        min_size (int, optional): x/y/z minimum value. Defaults to 0.

    Returns:
        Tuple[tf.Tensor]: length: len(scales)
    """
    pyramid = []
    for i in range(num_scales):
        re_data = resize_3d(data, scale_factor=scales[i])
        pyramid.append(re_data)
    return pyramid