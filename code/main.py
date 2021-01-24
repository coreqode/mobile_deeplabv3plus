import tensorflow as tf
from opts import get_parser
from trainer import trainer
from utils import start_tpu
from k_worst import k_worst
import os


def main(opts):
    """docstring for main"""
    if opts.mode == "colab_tpu":
        strategy = start_tpu()
        t = trainer(opts, strategy)
        t.train()

    elif opts.mode == "gpu":
        opts.shuffle_buffer = 64
        opts.val_batch_size = 32
        opts.train_batch_size = 32
        strategy = tf.distribute.get_strategy()
        t = trainer(opts, strategy)
        t.train()

    elif opts.mode == "debug":
        opts.shuffle_buffer = None
        opts.val_batch_size = 1
        opts.train_batch_size = 1
        strategy = tf.distribute.get_strategy()
        t = trainer(opts, strategy)
        t.train()


if __name__ == "__main__":
    opts = get_parser()
    num_data = opts.tfrecord_path.split('/')[-1].split('.')[0].split('_')[-1]
    opts.num_data = int(num_data)
    opts.weight_path =  opts.weight_path + opts.exp_name
    main(opts)
