import tensorflow as tf
import argparse
import tensorflow_addons as tfa
from data import create_dataset
import cv2
from PIL import Image
import math
import numpy as np
from loss import total_loss
from deeplabv3plus import deeplabv3
from callbacks import *
import json
from opts import get_parser
from progress.bar import Bar
import tensorflow_addons as tfa
from utils import get_weights


class trainer(object):
    """docstring for trainer"""

    def __init__(self, opts, strategy):
        self.opts = opts
        self.strategy = strategy
        with self.strategy.scope():
            if self.opts.model == "mobilenet":
                inp = tf.keras.layers.Input(
                    (self.opts.input_height, self.opts.input_width, 3)
                )
                out = deeplabv3(
                    inp,
                    (self.opts.input_height, self.opts.input_width, 3),
                    self.opts.num_classes,
                    8,
                )
                self.model = Model(inp, out)

            if self.opts.restart:
                try:
                    self.model.load_weights(self.opts.weight_path)
                except:
                    self.model = tf.keras.models.load_model(
                        self.opts.weight_path, custom_objects={"tf": tf}, compile=False
                    )

            if self.opts.mode != "colab_tpu":
                self.train_gen = create_dataset(self.opts, "train")
                self.val_gen = create_dataset(self.opts, "val")
            else:
                per_replica_train_batch_size = (
                    self.opts.train_batch_size // self.strategy.num_replicas_in_sync
                )
                per_replica_val_batch_size = (
                    self.opts.val_batch_size // self.strategy.num_replicas_in_sync
                )
                self.train_gen = self.strategy.experimental_distribute_datasets_from_function(
                    lambda _: create_dataset(
                        self.opts, "train", per_replica_train_batch_size
                    )
                )
                self.val_gen = self.strategy.experimental_distribute_datasets_from_function(
                    lambda _: create_dataset(
                        self.opts, "val", per_replica_val_batch_size
                    )
                )
                setattr(self.train_gen, "length", int(0.8 * self.opts.num_data))
                setattr(self.val_gen, "length", int(0.2 * self.opts.num_data))

            radam = tfa.optimizers.RectifiedAdam(lr=self.opts.learning_rate)
            self.optimizer = tfa.optimizers.Lookahead(
                radam, sync_period=6, slow_step_size=0.5
            )

            # lr_scheduler = tf.keras.experimental.CosineDecay(self.opts.learning_rate, 200)
            # self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.opts.learning_rate, name = 'Adam')
            # self.optimizer = tf.keras.optimizers.SGD(learning_rate= lr_scheduler, momentum=0.999, nesterov=True, name='SGD')
            self.train_loss = tf.keras.metrics.Mean(name="train_loss")
            self.val_loss = tf.keras.metrics.Mean(name="val_loss")
        self.loss_func = total_loss
        self.summary_writer = tf.summary.create_file_writer(
            self.opts.tensorboard_logs + self.opts.exp_name + "/logs"
        )
        self.early_stopping = EarlyStoppingAtMinLoss(self.opts, self.model, patience=3,)

    def train(self):
        """ Complete Training Loop """
        print("")
        print(f"Dataset Volume : {self.opts.num_data}")
        print(f"Trainset Volume : {self.train_gen.length}")
        print(f"Valset Volume : {self.val_gen.length}")
        print("")

        self.early_stopping.on_train_begin()
        for epoch in range(1, self.opts.epochs + 1):

            ######################## Training Loop ############################
            num_iterations = int(self.train_gen.length // self.opts.train_batch_size)

            if self.opts.mode == "colab_tpu":
                train_iterator = iter(self.train_gen)
                bar = Bar(f"Ep : {epoch} | Training :", max=num_iterations)
                for step in range(num_iterations):
                    self.tpu_train_step(train_iterator)
                    Bar.suffix = f"{step+1}/{num_iterations} | Total: {bar.elapsed_td:} | ETA: {bar.eta_td:} | train_loss: {self.train_loss.result().numpy()}"
                    bar.next()
                bar.finish()

            else:
                bar = Bar(f"Ep : {epoch} | Training :", max=num_iterations)
                for batch_idx, data in enumerate(self.train_gen):
                    inputs, targets = data
                    predictions = self.train_step(inputs, targets)
                    Bar.suffix = f"{batch_idx+1}/{num_iterations} | Total: {bar.elapsed_td:} | ETA: {bar.eta_td:} | train_loss: {self.train_loss.result().numpy()}"
                    bar.next()
                bar.finish()

            # step = np.int64(epoch)
            # step = tf.constant(step, dtype="int64")
            # self.dump_scalars(step, train=True)

            # TODO Add the dumping images function

            ############################# Valdiation Loop ##############################
            num_iterations = int(self.val_gen.length // self.opts.val_batch_size)

            if self.opts.mode == "colab_tpu":
                val_iterator = iter(self.val_gen)
                bar = Bar(f"Ep : {epoch} | Validation :", max=num_iterations)
                for step in range(num_iterations):
                    self.tpu_val_step(val_iterator)
                    Bar.suffix = f"{step+1}/{num_iterations} | Total: {bar.elapsed_td:} | ETA: {bar.eta_td:} | val_loss: {self.val_loss.result().numpy()}"
                    bar.next()
                bar.finish()

            else:
                for batch_idx, data in enumerate(self.val_gen):
                    predictions = self.val_step(data)
                    # if epoch % opts.dump_image_frq == 0:
                    #   if batch_idx < 5:
                    #     self.dump_images(predictions[0], targets[0], step, epoch, batch_idx, val=True)
                    Bar.suffix = f"{batch_idx+1}/{num_iterations} | Total: {bar.elapsed_td:} | ETA: {bar.eta_td:} | val_loss: {self.val_loss.result().numpy()}"
                    bar.next()
                bar.finish()

            # # self.dump_scalars(step, val = True)

            self.early_stopping.on_epoch_end(epoch, self.val_loss.result().numpy())
            self.train_loss.reset_states()
            self.val_loss.reset_states()
            tf.print("")

    @tf.function
    def tpu_train_step(self, iterator):
        self.strategy.run(self.train_step, args=(next(iterator)))

    @tf.function
    def tpu_val_step(self, iterator):
        self.strategy.run(self.val_step, args=(next(iterator)))

    @tf.function
    def train_step(self, inputs, targets):
        """ Training for one step"""
        with tf.GradientTape() as tape:
            predictions = self.model(inputs, training=True)
            loss = self.loss_func(self.opts, targets, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)

        if self.opts.mode == "colab_tpu":
            self.optimizer.apply_gradients(
                list(zip(gradients, self.model.trainable_variables))
            )
            self.train_loss.update_state(loss * self.strategy.num_replicas_in_sync)
        else:
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables)
            )
            self.train_loss.update_state(loss)
            return predictions

    @tf.function
    def val_step(self, inputs, targets):
        """ validation for one step"""
        predictions = self.model(inputs)
        loss = self.loss_func(self.opts, targets, predictions)
        if self.opts.mode == "colab_tpu":
            self.val_loss.update_state(loss * self.strategy.num_replicas_in_sync)
        else:
            self.val_loss.update_state(loss)
            return predictions
