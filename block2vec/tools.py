from pathlib import Path
from typing import List, Tuple

import numpy as np

from world.world import World, EmptyWrold


def create_embed_world(
        world_dir: Path,
        lims: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
        output_dir: Path
    ) -> Path:
    """export and save world in latent space

    Args:
        world_dir (Path): world dir
        lims (Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]): (xlim, ylim, zlim)
        output_dir (Path): the output dir of the world in latent space

    Returns:
        Path: the file path of world in latent space (np.ndarray)
    """
    # read block names
    blocks = []
    with open(output_dir / "metadata.tsv", "r", encoding="utf-8") as f:
        for line in f.readlines():
            blocks.append(line.strip())
    block2idx = {block: i for i, block in enumerate(blocks)}
    # read embeddings
    embeddings = np.load(output_dir / "embeddings.npy")

    # walk through world
    # world start from 0.0.0, read all blocks to embeddings array
    world = World(world_dir)
    xlim, ylim, zlim = lims
    xdim, ydim, zdim = xlim[1] - xlim[0] + 1, ylim[1] - ylim[0] + 1, zlim[1] - zlim[0] + 1
    embed_world = np.zeros([xdim, ydim, zdim, embeddings.shape[1]], dtype=np.float32)
    for x in range(xlim[0], xlim[1] + 1):
        for y in range(ylim[0], ylim[1] + 1):
            for z in range(zlim[0], zlim[1] + 1):
                block = world.get_block(x, y, z).id
                idx = block2idx[block]
                embed_world[x - xlim[0], y - ylim[0], z - zlim[0], :] = embeddings[idx]
    np.save(output_dir / "embed_world.npy", embed_world)
    return output_dir / "embed_world.npy"

def load_embed_world(output_dir: Path):
    embed_world = np.load(output_dir / "embed_world.npy")
    return embed_world

def load_embeddings(output_dir: Path) -> Tuple[List[str], np.ndarray]:
    """list of block names and embeddings in the same order

    Args:
        output_dir (Path): the dir to metadata.tsv and embeddings.npy

    Returns:
        Tuple[List[str], np.ndarray]: (block names, embeddings)
    """
    blocks = []
    with open(output_dir / "metadata.tsv", "r", encoding="utf-8") as f:
        for line in f.readlines():
            blocks.append(line.strip())
    embeddings = np.load(output_dir / "embeddings.npy")
    return blocks, embeddings

def export_world(
        embed_world: np.ndarray,
        blocks: List[str],
        embeddings: np.ndarray,
        empty_world_dir: Path,
        world_name: str,
        out_dir: Path
    ) -> Path:
    """transfer world (in the latent space) to world (in the geo space), and save it to out_dir

    Args:
        embed_world (np.ndarray): world (in the latent space)
        blocks (List[str]): block names
        embeddings (np.ndarray): embeddings
        empty_world_dir (Path): empty world in minecraft, just for export
        world_name (str): world name
        out_dir (Path): output dir

    Returns:
        Path: the path to the exported world
    """
    empty_world = EmptyWrold(empty_world_dir)
    for x in range(embed_world.shape[0]):
        for y in range(embed_world.shape[1]):
            for z in range(embed_world.shape[2]):
                # choose the block name according to the minimum l2 distance
                l2 = np.linalg.norm(embeddings - embed_world[x, y, z], axis=1)
                idx = np.argmin(l2)
                empty_world.add_block(x, y, z, blocks[idx])
    empty_world.zip(out_dir, world_name)
    return out_dir / f"{world_name}.zip"
