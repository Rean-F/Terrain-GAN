import math


def block2chunk(x, y, z):
    return (math.floor(x / 16), math.floor(z / 16))

def chunk2region(chunk_x, chunk_z):
    return (math.floor(chunk_x / 32), math.floor(chunk_z / 32))

def block2region(x, y, z):
    return (math.floor(x / 32 / 16), math.floor(z / 32 / 16))

def block_in_chunk(x, y, z):
    return (x % 16, y, z % 16)

def idx2block(idx):
    y = math.floor(idx // 256)
    z = math.floor((idx - y * 256) / 16)
    x = idx - y * 256 - z * 16
    return x, y, z

def block2idx(x, y, z):
    idx = x + z * 16 + y * 256
    return idx
