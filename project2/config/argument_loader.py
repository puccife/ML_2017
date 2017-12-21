import json
import os
import tensorflow as tf

CONFIG_DIR = 'config/'
config_json = 'configuration.json'

class ArgumentLoader:

    def get_configuration(self):
        """
        Function used to read the configuration file and create the tf.FLAGS 
        :return: FLAGS with properties used for training.
        """
        FLAGS = tf.flags.FLAGS
        with open(os.path.join(CONFIG_DIR, config_json)) as data_file:
            configuration = json.load(data_file)
            tf.flags.DEFINE_integer("hidden_size", int(configuration["hidden_size"]), "Size of LSTM hidden layer.")
            tf.flags.DEFINE_integer("memory_size", int(configuration["memory_size"]), "The number of memory slots.")
            tf.flags.DEFINE_integer("word_size", int(configuration["word_size"]), "The width of each memory slot.")
            tf.flags.DEFINE_integer("num_write_heads", int(configuration["num_write_heads"]), "Number of memory write heads.")
            tf.flags.DEFINE_integer("num_read_heads", int(configuration["num_read_heads"]), "Number of memory read heads.")
            tf.flags.DEFINE_integer("clip_value", int(configuration["clip_value"]),
                                    "Maximum absolute value of controller and dnc outputs.")
            tf.flags.DEFINE_float("max_grad_norm", float(configuration["max_grad_norm"]), "Gradient clipping norm limit.")
            tf.flags.DEFINE_float("learning_rate", float(configuration["learning_rate"]), "Optimizer learning rate.")
            tf.flags.DEFINE_float("final_learning_rate", float(configuration["final_learning_rate"]), "Optimizer final learning rate.")
            tf.flags.DEFINE_float("optimizer_epsilon", float(configuration["optimizer_epsilon"]),
                                  "Epsilon used for RMSProp optimizer.")
            tf.flags.DEFINE_string("dataset_neg",configuration["dataset_neg"],"Dataset of negative tweets.")
            tf.flags.DEFINE_string("dataset_pos",configuration["dataset_pos"],"Dataset of positive tweets.")
            tf.flags.DEFINE_string("dataset_test",configuration["dataset_test"],"Dataset of testing tweets.")
            tf.flags.DEFINE_string("glove_dir", configuration["glove_dir"], "The location of GloVe pretrained model.")
            tf.flags.DEFINE_boolean("random", configuration["random"], "True if you want to randomized the rewiew to choose")
            tf.flags.DEFINE_boolean("seed", int(configuration["seed"]), "The seed that you want to set")
            tf.flags.DEFINE_float("ratio", float(configuration["ratio"]), "The ratio of training/testing splitting")
            tf.flags.DEFINE_integer("batch_size", int(configuration["batch_size"]), "Batch size for training.")
            tf.flags.DEFINE_integer("max_lenght", int(configuration["max_lenght"]), "Max number of word of the review.")
            tf.flags.DEFINE_integer("word_dimension", int(configuration["word_dimension"]), "The number of dimension of W2V")
            tf.flags.DEFINE_integer("num_classes", int(configuration["num_classes"]),
                                    "Number of classes")
            tf.flags.DEFINE_integer("num_testing_iterations",int( configuration["num_testing_iterations"]),
                                    "Number of iterations to train for.")
            tf.flags.DEFINE_integer("num_epochs", int(configuration["num_epochs"]),
                                    "Number of epoch.")
            tf.flags.DEFINE_integer("report_interval", int(configuration["report_interval"]),
                                    "Iterations between reports (samples, valid loss).")
            tf.flags.DEFINE_string("checkpoint_dir", configuration["checkpoint_dir"],
                                   "Checkpointing directory.")
            tf.flags.DEFINE_integer("checkpoint_interval", int(configuration["checkpoint_interval"]),
                                    "Checkpointing step interval.")
            tf.flags.DEFINE_integer("total_samples", int(configuration["total_samples"]),
                                    "Number of samples.")
        return FLAGS
