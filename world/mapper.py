from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
import webcolors

from world.block_colors import block_colors

class Mapper:
    """地图绘制
    """
    def __init__(self) -> None:
        self.points = []
        self.colors = []
        self.air_idxs = []

    def add_block(self, x, y, z, block: str) -> None:
        self.points.append([x, y, z])
        colorname = block_colors.get(block, "black")
        color = webcolors.name_to_rgb(colorname)
        self.colors.append([color.red, color.green, color.blue])

        if block == "air":
            self.air_idxs.append(len(self.points) - 1)
        return

    def draw(self, map_dir: Path, fname="draw_3d.png") -> None:
        points = np.array(self.points, dtype=np.int32)
        colors = np.array(self.colors, dtype=np.float32) / 255.0
        points = np.delete(points, self.air_idxs, axis=0)
        colors = np.delete(colors, self.air_idxs, axis=0)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(points[:, 0], points[:, 2], points[:, 1], c=colors)
        plt.savefig(str(map_dir / fname))

    def render(self, map_dir: Path) -> None:
        pass