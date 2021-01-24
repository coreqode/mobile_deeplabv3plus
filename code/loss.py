import tensorflow.keras.backend as K
import tensorflow as tf


def total_loss(opts, y_true, y_pred):

    def boundary_loss_v1(y_true, y_pred):
      scce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
      ground_truth = y_true[...,0]
      ground_truth = tf.one_hot(ground_truth, depth = 15)
      boundary_mask = tf.cast(y_true[...,1], tf.float32)/255.0
      loss = scce(ground_truth, y_pred)
      boundary_loss = loss * boundary_mask
      loss = loss + 5 * boundary_loss
      return loss

    def wce(y_true, y_pred):
      scce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
      ground_truth = y_true[...,0]
      ground_truth = tf.one_hot(ground_truth, depth = 15)
      loss = scce(ground_truth, y_pred)
      return loss

    loss = boundary_loss_v1(y_true, y_pred)
    if opts.mode ==  'colab_tpu':
      loss = tf.reduce_sum(loss) / (opts.input_height * opts.input_width * opts.train_batch_size) #https://www.tensorflow.org/tutorials/distribute/custom_training#define_the_loss_function
    else:
      loss =  tf.reduce_mean(loss)
    return loss

