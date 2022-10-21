try:
    import tensorflow as tf
    TENSORBOARD = True
except ImportError:
    print('tensorflow not found')
    TENSORBOARD = False



class Logger:
    def __init__(self, log_dir=None):
        if TENSORBOARD:
            self.writer = tf.summary.create_file_writer(log_dir)


    def scalar_summary(self, tag, value, step):
        if TENSORBOARD:
            with self.writer.as_default():
                tf.summary.scalar(tag, value, step=step)
                self.writer.flush()

