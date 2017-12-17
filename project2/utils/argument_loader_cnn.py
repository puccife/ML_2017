import json
import os
import tensorflow as tf

CONFIG_DIR = 'config/'
config_json = 'configuration_cnn.json'

class ArgumentLoader_cnn:

    def get_configuration(self):
        FLAGS = tf.flags.FLAGS
        with open(os.path.join(CONFIG_DIR, config_json), encoding="utf-8", errors='ignore') as data_file:
            configuration = json.load(data_file)
            tf.flags.DEFINE_integer("num_filters", int(configuration["num_filters"]), "Size of last hidden layer.")
            tf.flags.DEFINE_integer("hidden_size", int(configuration["hidden_size"]), "Numbr of filters")
            tf.flags.DEFINE_string("dataset_neg",configuration["dataset_neg"],"Dataset of negative tweets.")
            tf.flags.DEFINE_string("dataset_pos",configuration["dataset_pos"],"Dataset of positive tweets.")
            tf.flags.DEFINE_string("glove_dir", configuration["glove_dir"], "The location of GloVe pretrained model.")
            tf.flags.DEFINE_boolean("seed", int(configuration["seed"]), "The seed that you want to set")
            tf.flags.DEFINE_float("ratio", float(configuration["ratio"]), "The ratio of training/testing splitting")
            tf.flags.DEFINE_integer("batch_size", int(configuration["batch_size"]), "Batch size for training.")
            tf.flags.DEFINE_integer("max_length", int(configuration["max_length"]), "Max number of word of the review.")
            tf.flags.DEFINE_integer("word_dimension", int(configuration["word_dimension"]), "The number of dimension of W2V")
            tf.flags.DEFINE_integer("num_epochs", int(configuration["num_epochs"]),
                                    "Number of epoch.")
            tf.flags.DEFINE_integer("steps_per_epoch", int(configuration["steps_per_epoch"]), "Sample size / batch size for training set.")
            tf.flags.DEFINE_integer("validation_step", int(configuration["validation_step"]), "Sample size / batch size for validation set.")
        return FLAGS
