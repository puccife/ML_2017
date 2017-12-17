import time
import json
import random
import math

import tensorflow as tf
import operator

from utils.manipulator_cnn import DatasetManipulator_cnn
from utils.pretrained_glove_cnn import GloveTrainer_cnn
from keras.callbacks import ModelCheckpoint

import numpy as np
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout
from keras.layers import GRU, LSTM
from keras.models import Model
from keras.layers.merge import Concatenate
from keras.preprocessing import sequence
import re
import itertools
from collections import Counter
from utils.preprocessing import clean_tweets
import os


class CNNTrainer:

    FLAGS = None
    x_train = None
    x_test = None
    y_train = None
    y_test = None
    dm_cnn= None

    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.__init_model()

    def generator_validator(self):

        while 1:
            for j in range(int(len(self.x_test)/self.FLAGS.batch_size)):

                x_test_embeddings = self.gt_cnn.manipulate_dataset(self.x_test[j*self.FLAGS.batch_size:((j+1)*self.FLAGS.batch_size)],self.word_embeddings)
                y_test_batch = self.y_test[(j*self.FLAGS.batch_size):((j+1)*self.FLAGS.batch_size)]

                if(j==(int(len(self.x_test)/self.FLAGS.batch_size)-1)):
                    print ("Validation Epoch Done")
                    print(x_test_embeddings.shape)
                    print(y_test_batch.shape)
                yield x_test_embeddings, y_test_batch

    def generator(self):

        while 1:
            for i in range(int(len(self.x_train)/self.FLAGS.batch_size)):
                x_train_embeddings = self.gt_cnn.manipulate_dataset(self.x_train[i*self.FLAGS.batch_size:((i+1)*self.FLAGS.batch_size)],self.word_embeddings)
                y_train_batch = self.y_train[(i*self.FLAGS.batch_size):((i+1)*self.FLAGS.batch_size)]

                if(i == (int(len(self.x_train)/self.FLAGS.batch_size)-1)):
                    print ("Training Epoch Done ")
                    print(x_train_embeddings.shape)
                    print(y_train_batch.shape)

                yield x_train_embeddings, y_train_batch

    def __init_model(self):

        self.gt_cnn = GloveTrainer_cnn(tweet_length=self.FLAGS.max_length,vector_size=self.FLAGS.word_dimension, glove_dir=self.FLAGS.glove_dir)
        self.word_embeddings = self.gt_cnn.generate_word_embeddings()

        self.dm_cnn = DatasetManipulator_cnn(tweet_length=self.FLAGS.max_length,positive_url=self.FLAGS.dataset_pos,negative_url=self.FLAGS.dataset_neg)
        sentences, labels = self.dm_cnn.load_data_and_labels()
        sentences_padded = self.dm_cnn.pad_sentences(sentences)

        x = sentences_padded
        y = np.array(labels)

        y = y.argmax(axis=1)

        self.x_train,self.x_test,self.y_train,self.y_test = self.dm_cnn.split_and_shuffle(x,y,ratio=self.FLAGS.ratio,seed=self.FLAGS.seed)

        print("x_train shape:", len(self.x_train))
        print("x_test shape:", len(self.x_test))
        print("y_train shape:", len(self.y_train))
        print("y_test shape:", len(self.y_test))


        self.run_model()

    def run_model(self):

        print("Running model")

        dropout_prob = (0.1, 0.5)
        filter_sizes = (3,5,3,3,5,3,3)

        input_shape = (self.FLAGS.max_length, self.FLAGS.word_dimension)
        model_input = Input(shape=input_shape)

        z = model_input
        #z = Dropout(dropout_prob[0])(z)

        conv_blocks = []
        for sz in filter_sizes:
            conv = Conv1D(filters=self.FLAGS.num_filters,
                                 kernel_size=sz,
                                 padding="valid",
                                 activation="relu",
                                 strides=1)(z)

            conv = MaxPooling1D(pool_size=2)(conv)
            conv = LSTM(128)(conv)

            #conv = Flatten()(conv)
            conv_blocks.append(conv)

        z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

        z = Dropout(dropout_prob[1])(z)
        z = Dense(self.FLAGS.hidden_size, activation="relu")(z)
        model_output = Dense(1, activation="sigmoid")(z)

        model = Model(model_input, model_output)
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

        # checkpoint
        filepath="best_weights.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        model.fit_generator(self.generator(),steps_per_epoch=self.FLAGS.steps_per_epoch, epochs=self.FLAGS.num_epochs,validation_data= self.generator_validator(),validation_steps=self.FLAGS.validation_step, verbose=2,callbacks=callbacks_list)
        model.save('my_test_model.h5')
