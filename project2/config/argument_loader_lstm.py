import json
import os
import tensorflow as tf

CONFIG_DIR = 'config/'
config_json = 'configuration_lstm.json'

class ArgumentLoaderLstm:

    def get_configuration(self):
        FLAGS = tf.flags.FLAGS
        with open(os.path.join(CONFIG_DIR, config_json), encoding="utf-8", errors='ignore') as data_file:
            configuration = json.load(data_file)
            tf.flags.DEFINE_integer("embedding_dim", int(configuration["embedding_dim"]), "Size of the word embedding")
            tf.flags.DEFINE_string("model_name",configuration["model_name"],"Name of the model")
            tf.flags.DEFINE_string("model_path",configuration["model_path"],"Path of the model")
            tf.flags.DEFINE_string("embedding_dir", configuration["embedding_dir"], "embedding directory")
            tf.flags.DEFINE_string("train_data_file_pos", str(configuration["train_data_file_pos"]), "training dataset positive")
            tf.flags.DEFINE_string("train_data_file_neg", str(configuration["train_data_file_neg"]), "training dataset negative")
            tf.flags.DEFINE_string("test_data_file", str(configuration["test_data_file"]), "testing dataset")
            tf.flags.DEFINE_integer("max_sequence_length", int(configuration["max_sequence_length"]), "Max number of word of the tweet.")
            tf.flags.DEFINE_integer("max_nb_words", int(configuration["max_nb_words"]), "Maximum number of words into embedding.")
            tf.flags.DEFINE_float("validation_split", float(configuration["validation_split"]), "Validation split.")
        return FLAGS
