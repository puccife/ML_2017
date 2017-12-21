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
            tf.flags.DEFINE_integer("data_dir", int(configuration["data_dir"]), "directory of the data")
            tf.flags.DEFINE_integer("embedding_dim", int(configuration["embedding_dim"]), "Size of the word embedding")
            tf.flags.DEFINE_string("model_name",configuration["model_name"],"Name of the model")
            tf.flags.DEFINE_string("model_path",configuration["model_path"],"Path of the model")
            tf.flags.DEFINE_string("embedding_dir", configuration["embedding_dir"], "embedding directory")
            tf.flags.DEFINE_boolean("train_data_file_pos", int(configuration["train_data_file_pos"]), "training dataset positive")
            tf.flags.DEFINE_float("train_data_file_neg", float(configuration["train_data_file_neg"]), "training dataset negative")
            tf.flags.DEFINE_integer("test_data_file", int(configuration["test_data_file"]), "testing dataset")
            tf.flags.DEFINE_integer("max_sequence_length", int(configuration["max_sequence_length"]), "Max number of word of the tweet.")
            tf.flags.DEFINE_integer("max_nb_words", int(configuration["max_nb_words"]), "Maximum number of words into embedding.")
            tf.flags.DEFINE_integer("validation_split", int(configuration["validation_split"]), "Validation split.")
        return FLAGS
