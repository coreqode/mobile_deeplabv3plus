import numpy as np
import tensorflow as tf 

class EarlyStoppingAtMinLoss(object):
    def __init__(self, opts,  model , patience=0, ):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience
        self.best_weights = None
        self.save_path = opts.save_path + opts.exp_name + '/weights/'
        self.exp_name = opts.exp_name
        self.save_freq = opts.save_freq
        self.model = model

    def on_train_begin(self, val_loss=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf

    def on_epoch_end(self, epoch, val_loss=None):
        current = val_loss
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            self.best_weights = self.model.get_weights()
            if epoch % self.save_freq == 0:
                self.model.save(self.save_path + self.exp_name, save_format = 'tf')
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, val_loss=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))
            self.model.save(self.save_path + self.exp_name, save_format = 'tf')


class CustomLearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, opts, schedule):
        super(CustomLearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.lr_schedule = opts.lr_schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        scheduled_lr = self.schedule(epoch, lr, self.lr_schedule)
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        print("\nEpoch %05d: Learning rate is %6.4f." % (epoch, scheduled_lr))


def lr_schedule(epoch, lr, LR_SCHEDULE):
    if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
        return lr
    for i in range(len(LR_SCHEDULE)):
        if epoch == LR_SCHEDULE[i][0]:
            return LR_SCHEDULE[i][1]
    return lr



