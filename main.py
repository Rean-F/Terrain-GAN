import argparse

import tensorflow as tf

from params import *
from block2vec.train import train as block2vec_train
from block2vec.tools import create_embed_world, export_world, load_embed_world, load_embeddings
from singan_3d.inference import Inferencer
from singan_3d.tools import create_pyramid
from singan_3d.train import Trainer
from world.world import World
from world.mapper import Mapper


args_parser = argparse.ArgumentParser(description="Minecraft-GAN")
args_parser.add_argument(
    "-t",
    "--task",
    type=str,
    choices=[
        "draw_world", "block2vec_train", "create_embed_world",
        "create_pyramid", "singan_train", "random_generate", "space_transfer"
    ]
)
args = args_parser.parse_args()

world_dir = CommonParams.world_dir
xlim, ylim, zlim = CommonParams.xlim, CommonParams.ylim, CommonParams.zlim
map_dir = CommonParams.map_dir

if args.task == "draw_world":
    world = World(world_dir)
    mapper = Mapper()
    for x in range(xlim[0], xlim[1] + 1):
        for y in range(ylim[0], ylim[1] + 1):
            for z in range(zlim[0], zlim[1] + 1):
                mapper.add_block(x, y, z, world.get_block(x, y, z).id)
    mapper.draw(map_dir)

if args.task == "block2vec_train":
    """train block2vec model and export embeddings.
    """
    block2vec_train(
        world_dir,
        lims=[xlim, ylim, zlim],
        embed_dim=Block2VecParams.embed_dim,
        epochs=Block2VecParams.epochs,
        log_dir=Block2VecParams.log_dir,
        model_dir=Block2VecParams.model_dir,
        output_dir=Block2VecParams.out_dir
    )

if args.task == "create_embed_world":
    """generate world in latent sapce.
    """
    create_embed_world(
        world_dir,
        lims=[xlim, ylim, zlim],
        output_dir=Block2VecParams.out_dir
    )

if args.task == "create_pyramid":
    """export downsampled world in latent space.
    """
    embed_world = load_embed_world(Block2VecParams.out_dir)
    embed_world = tf.convert_to_tensor(embed_world)
    embed_world = tf.expand_dims(embed_world, axis=0)
    blocks, embeddings = load_embeddings(Block2VecParams.out_dir)
    pyramid = create_pyramid(
        embed_world,
        num_scales=SinGanParams.num_scales,
        scales=SinGanParams.scales
    )

    for scale, world in enumerate(pyramid):
        export_world(world, blocks, embeddings, ExportParams.empty_world_dir, f"pyramid_{scale}", SinGanParams.out_dir)
    pass

if args.task == "singan_train":
    """train singan generate model.
    """
    embed_world = load_embed_world(Block2VecParams.out_dir)
    blocks, embeddings = load_embeddings(Block2VecParams.out_dir)
    embed_world = tf.convert_to_tensor(embed_world)
    embed_world = tf.expand_dims(embed_world, axis=0)

    trainer = Trainer(
        channels_num=Block2VecParams.embed_dim,
        num_scales=SinGanParams.num_scales,
        scales=SinGanParams.scales,
        learning_rate=SinGanParams.lr,
        n_iters=SinGanParams.n_iters,
        noise_w=SinGanParams.noise_w,
        model_dir=SinGanParams.model_dir,
        log_dir=SinGanParams.log_dir,
        out_dir=SinGanParams.out_dir
    )
    for scale, step, real, fake_rand in trainer.train(embed_world):
        world_name = f"fake_rand_{scale}_{step}"
        export_world(fake_rand.numpy()[0], blocks, embeddings, ExportParams.empty_world_dir, world_name, SinGanParams.out_dir)
    pass

if args.task == "random_generate":
    """singan inference
    """
    embed_world = load_embed_world(Block2VecParams.out_dir)
    blocks, embeddings = load_embeddings(Block2VecParams.out_dir)
    embed_world = tf.convert_to_tensor(embed_world)
    embed_world = tf.expand_dims(embed_world, axis=0)

    inferencer = Inferencer(
        channels_num=Block2VecParams.embed_dim,
        num_scales=SinGanParams.num_scales,
        scales=SinGanParams.scales,
        model_dir=SinGanParams.model_dir,
        out_dir=SinGanParams.out_dir
    )

    for i in range(3):
        rand_world = inferencer.random_generate(embed_world)
        world_name = f"rand_generate_{i}"
        export_world(rand_world.numpy()[0], blocks, embeddings, ExportParams.empty_world_dir, world_name, SinGanParams.out_dir)
    # random size generate
    rand_world = inferencer.rand_size_generate(dims=[150, 25, 150])
    world_name = "rand_2x_world"
    export_world(rand_world.numpy()[0], blocks, embeddings, ExportParams.empty_world_dir, world_name, SinGanParams.out_dir)