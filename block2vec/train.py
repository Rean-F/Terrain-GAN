from collections import OrderedDict
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import optimizers

from block2vec.dataset import Block2VecDataset
from block2vec.model import SkipGram
from block2vec.tools import create_embed_world


def skip_gram_loss(context: tf.Tensor, output: tf.Tensor) -> tf.Tensor:
    """custom skip-gram loss

    Args:
        context (tf.Tensor): context embeddings, shape: [batch_size, (win_size + 1) ** 3 - 1]
        output (tf.Tensor): shape: [batch_size, onehot-dim]

    Returns:
        tf.Tensor: loss
    """
    win_size = context.shape[1]
    # calculate cross entropy with every neighbor
    losses = []
    for i in tf.range(win_size):
        word_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=context[:, i], logits=output)
        losses.append(word_loss)
    losses = tf.stack(losses)
    # mean of all cross entropy
    loss = tf.reduce_mean(losses)
    return loss

def train(
        world_dir: Path,
        lims: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
        embed_dim: int,
        epochs: int,
        log_dir: Path,
        model_dir: Path,
        output_dir: Path
    ) -> None:
    """start training

    Args:
        world_dir (Path): minecraft world dir
        lims (Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]): (xlim, ylim, zlim)
        embed_dim (int): embedding's dim in the latent space
        epochs (int): epochs
        log_dir (Path): log dir for block2vec
        model_dir (Path): model dir for block2vec
        output_dir (Path): output dir for block2vec
    """
    dataset = Block2VecDataset(world_dir, lims, win_radius=1)

    onehot_dim = len(dataset.get_block2id())
    model = SkipGram(input_dim=onehot_dim, embed_dim=embed_dim)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=3e-4),
        loss=skip_gram_loss,
        run_eagerly=True
    )
    
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=str(log_dir),
        histogram_freq=1,
        write_graph=True,
        update_freq="epoch",
        profile_batch=0,
        embeddings_freq=1
    )

    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=str(model_dir / "block2vec.mod"),
        save_weights_only=False
    )

    class SaveEmbeddingsCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch: int, logs=None):
            """save block names and embeddings

            Args:
                epoch (int): epoch
                logs ([type], optional): Defaults to None
            """
            block2id = dataset.get_block2id()
            blocks = []
            ids = np.zeros([len(block2id), 1], dtype=np.int32)
            for i, (block, id) in enumerate(block2id.items()):
                blocks.append(block)
                ids[i, 0] = id
            # calculate embedding layer output
            embeddings = model.embedding(tf.convert_to_tensor(ids))
            embeddings = tf.squeeze(embeddings, axis=1).numpy()
            # save embeddings.npy and metadata.tsv
            with open(output_dir / "embeddings.npy", "wb") as f:
                np.save(f, embeddings)
            with open(output_dir / "metadata.tsv", "w", encoding="utf-8") as f:
                f.writelines("\n".join(blocks))
            return
        
        def on_train_end(self, logs=None):
            create_embed_world(world_dir, lims, output_dir)
    
    # set batch size to 2048
    model.fit(
        dataset.tf_dataset().batch(2048).prefetch(10),
        epochs=epochs,
        callbacks=[tensorboard_callback, checkpoint_callback, SaveEmbeddingsCallback()]
    )
    