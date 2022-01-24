import math
import random
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple
from itertools import product

import tensorflow as tf

from world.world import World

class ItemGenerator:
    """generator of tf dataset
    """
    def __init__(
        self,
        world_dir: Path,
        lims: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
        win_radius: int
    ) -> None:
        """initialize

        Args:
            world_dir (Path): minecraft world dir
            lims (Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]): lims of x.y.z
            win_radius (int): window size of skip-gram model
        """
        self.world = World(world_dir)
        self.xlim = lims[0]
        self.ylim = lims[1]
        self.zlim = lims[2]
        self.win_radius = win_radius
        # block counts in the world
        self.block_freq = OrderedDict()
        # block subsample prob, exp. air has very small prob
        self.block_subsample_prob = OrderedDict()
        # one-hot
        self.block2id = OrderedDict()
        self.id2block = OrderedDict()
        self._init_vocab()

    def _init_vocab(self) -> None:
        """walk thrhough the world and count the blocks then calculate the sub-sample probability.
        """
        for (x, y, z) in product(
            range(self.xlim[0], self.xlim[1] + 1),
            range(self.ylim[0], self.ylim[1] + 1),
            range(self.zlim[0], self.zlim[1] + 1)
        ):
            name = self.world.get_block(x, y, z).id
            self.block_freq[name] = self.block_freq.get(name, 0) + 1
        for block in self.block_freq:
            block_id = len(self.block2id)
            self.block2id[block] = block_id
            self.id2block[block_id] = block
        freq_sum = sum(self.block_freq.values())
        for block, freq in self.block_freq.items():
            f = freq / freq_sum
            self.block_subsample_prob[block] = math.sqrt(f / 0.001 + 1) * (0.001 / f)

    def _neighbors(self, x: int, y: int, z: int) -> List[str]:
        """get neighbors (in win-size) around x.y.z coord.

        Args:
            x (int): x
            y (int): y
            z (int): z

        Returns:
            List[str]: list of block's name in win-size around x.y.z
        """
        lims = [[c - self.win_radius, c + self.win_radius] for c in [x, y, z]]
        neighbors = []
        for (_x, _y, _z) in product(
            range(lims[0][0], lims[0][1] + 1),
            range(lims[1][0], lims[1][1] + 1),
            range(lims[2][0], lims[2][1] + 1)
        ):
            if not (_x == x and _y == y and _z == z):
                name = self.world.get_block(_x, _y, _z).id
                neighbors.append(name)
        return neighbors

    def _subsample(self, x: int, y: int, z: int) -> Tuple[int, List[int]]:
        """get one sample (block, list of blocks)

        Args:
            x (int): x
            y (int): y
            z (int): z

        Returns:
            Tuple[int, List[int]]: _1: x.y.z block one-hot id, _2: neighbor blocks one-hot ids
        """
        block = self.world.get_block(x, y, z).id
        if random.uniform(0, 1) <= self.block_subsample_prob[block]:
            neighbors = self._neighbors(x, y, z)
            return self.block2id[block], [self.block2id[nei] for nei in neighbors]
        # if value larger than subsumple prob, them random subsample again.
        else:
            padding = self.win_radius
            rand_x = random.randint(self.xlim[0] + padding, self.xlim[1] + 1 - padding)
            rand_y = random.randint(self.ylim[0] + padding, self.ylim[1] + 1 - padding)
            rand_z = random.randint(self.zlim[0] + padding, self.zlim[1] + 1 - padding)
            return self._subsample(rand_x, rand_y, rand_z)

    def __call__(self):
        padding = self.win_radius
        for (x, y, z) in product(
            range(self.xlim[0] + padding, self.xlim[1] + 1 - padding),
            range(self.ylim[0] + padding, self.ylim[1] + 1 - padding),
            range(self.zlim[0] + padding, self.zlim[1] + 1 - padding)
        ):
            block_id, neighbors_id = self._subsample(x, y, z)
            yield [block_id], neighbors_id


class Block2VecDataset:
    def __init__(
            self,
            world_dir: Path,
            lims: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
            win_radius: int
        ) -> None:
        """initialize

        Args:
            world_dir (Path): minecraft world dir
            lims (Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]): lims of x.y.z
            win_radius (int): window size of skip-gram model
        """
        self.generator = ItemGenerator(world_dir, lims, win_radius)
        self.world = self.generator.world
        self.xlim = self.generator.xlim
        self.ylim = self.generator.ylim
        self.zlim = self.generator.zlim
        self.output_types = (tf.int32, tf.int32)
        self.output_shapes = (tf.TensorShape([1,]), tf.TensorShape([(2*win_radius+1)**3-1,]))

    def tf_dataset(self) -> tf.data.Dataset:
        """get tensorflow dataset to train

        Returns:
            tf.data.Dataset: tf dataset
        """
        return tf.data.Dataset.from_generator(
            self.generator,
            output_types=self.output_types,
            output_shapes=self.output_shapes
        )

    def get_block2id(self):
        return self.generator.block2id

    def get_id2block(self):
        return self.generator.id2block

    def get_block_freq(self):
        return self.generator.block_freq
