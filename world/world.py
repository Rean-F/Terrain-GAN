from pathlib import Path
import shutil
import string
from typing import Dict, Tuple

import anvil

from world.tools import block2chunk, block2region, chunk2region, block_in_chunk


class World:
    """Minecraft地图信息
    """
    def __init__(self, world_dir: Path) -> None:
        if not world_dir.is_dir():
            raise FileNotFoundError(f"No such world dir: {world_dir}")

        self.world_dir = world_dir
        self.chunks: Dict[Tuple[int, int], anvil.Chunk] = {}

    def _load_chunk(self, chunk_x: int, chunk_z: int) -> anvil.Chunk:
        """加载区块信息(16*16 blocks)

        Args:
            chunk_x (int): 区块坐标x
            chunk_z (int): 区块坐标z

        Returns:
            anvil.Chunk: 区块信息
        """
        region_x, region_z = chunk2region(chunk_x, chunk_z)
        region_fname = f"r.{region_x}.{region_z}.mca"
        region_path = self.world_dir / "region" / region_fname
        chunk = anvil.Chunk.from_region(str(region_path), chunk_x, chunk_z)
        return chunk

    def get_block(self, x: int, y: int, z: int) -> anvil.Block:
        """返回xyz坐标对应的Block对象

        Args:
            x (int): block世界坐标x
            y (int): block世界坐标y
            z (int): block世界坐标z

        Returns:
            anvil.Block: xyz世界坐标对应的Block
        """
        chunk_x, chunk_z = block2chunk(x, y, z)
        chunk_pos = (chunk_x, chunk_z)
        if chunk_pos not in self.chunks:
            chunk = self._load_chunk(chunk_x, chunk_z)
            self.chunks[chunk_pos] = chunk
        chunk = self.chunks[chunk_pos]
        block = chunk.get_block(*(block_in_chunk(x, y, z)))
        return block


class EmptyWrold:
    def __init__(self, empty_world_dir: Path) -> None:
        if not empty_world_dir.is_dir():
            raise FileNotFoundError(f"No such empty world dir: {empty_world_dir}")
        
        self.world_dir = empty_world_dir
        self.regions: Dict[Tuple[int, int], anvil.EmptyRegion] = {}
        self.clean()

    def add_block(self, x, y, z, block_name: str):
        region_x, region_z = block2region(x, y, z)
        region_pos = (region_x, region_z)
        if region_pos not in self.regions:
            self.regions[region_pos] = anvil.EmptyRegion(region_x, region_z)
        region = self.regions[region_pos]

        block = anvil.Block("minecraft", block_name)
        region.set_block(block, x, y, z)

    def save(self):
        for pos, region in self.regions.items():
            with open(str(self.world_dir / "region" / f"r.{pos[0]}.{pos[1]}.mca"), "wb") as f:
                region.save(f)
        return

    def zip(self, out_dir: Path, world_name: str):
        self.save()
        shutil.make_archive(out_dir / world_name, "zip", self.world_dir)

    def clean(self):
        for path in (self.world_dir / "region").iterdir():
            path.unlink()
        return
        

