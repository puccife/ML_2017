from preprocessing.manipulator_cnn import DatasetManipulator_cnn
from word_embedding_model.pretrained_glove_cnn import GloveTrainer_cnn
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import numpy as np
from keras.layers import Dense, Input
from keras.layers import Conv1D, MaxPooling1D, Dropout ,SpatialDropout1D
from keras.layers import LSTM
from keras.models import Model
from keras.layers.merge import Concatenate


class CNNTrainer:

    FLAGS = None
    x_train = None
    x_test = None
    y_train = None
    y_test = None
    dm_cnn= None

    def __init__(self, FLAGS):
        """
        Initialize the CNN trainer
        :param FLAGS: Parameters of the CNN
        """
        self.FLAGS = FLAGS
        self.__init_model()

    def generator_validator(self):
        """
        A generator for the validation set used to pass a batch of tweets
        to the model to avoid memory errors.
        """
        # Infinite loop to keep feeding the model.
        while 1:
            # Number of batches
            for j in range(int(len(self.x_test)/self.FLAGS.batch_size)):
                # Create the embeddings for each batch
                x_test_embeddings = self.gt_cnn.manipulate_dataset(self.x_test[j*self.FLAGS.batch_size:((j+1)*self.FLAGS.batch_size)],self.word_embeddings)
                # Get the labels of the validating tweets.
                y_test_batch = self.y_test[(j*self.FLAGS.batch_size):((j+1)*self.FLAGS.batch_size)]

                if(j==(int(len(self.x_test)/self.FLAGS.batch_size)-1)):
                    print ("Validation Epoch Done")
                    print(x_test_embeddings.shape)
                    print(y_test_batch.shape)
                # passing it the model
                yield x_test_embeddings, y_test_batch

    def generator(self):
        """
        A generator for the training set used to pass a batch of tweets
        to the model to avoid memory errors.
        """
        # Infinite loop to keep feeding the model.
        while 1:
            # Number of batches
            for i in range(int(len(self.x_train)/self.FLAGS.batch_size)):
                # Create the embeddings for each batch
                x_train_embeddings = self.gt_cnn.manipulate_dataset(self.x_train[i*self.FLAGS.batch_size:((i+1)*self.FLAGS.batch_size)],self.word_embeddings)
                #  Get the labels of the training tweets.
                y_train_batch = self.y_train[(i*self.FLAGS.batch_size):((i+1)*self.FLAGS.batch_size)]

                if(i == (int(len(self.x_train)/self.FLAGS.batch_size)-1)):
                    print ("Training Epoch Done ")
                    print(x_train_embeddings.shape)
                    print(y_train_batch.shape)
                # passing it the model
                yield x_train_embeddings, y_train_batch

    def __init_model(self):
        """
        Initializing the model by generating the word embeddings, loading and preprocessing the dataset, and then splitting and
        shuffling the tweets into a training set and a validation set.
        """
        # Creating the word embeddings
        self.gt_cnn = GloveTrainer_cnn(tweet_length=self.FLAGS.max_length,vector_size=self.FLAGS.word_dimension, glove_dir=self.FLAGS.glove_dir)
        self.word_embeddings = self.gt_cnn.generate_word_embeddings()

        # Loading and preprocessing the dataset
        self.dm_cnn = DatasetManipulator_cnn(tweet_length=self.FLAGS.max_length,positive_url=self.FLAGS.dataset_pos,negative_url=self.FLAGS.dataset_neg)
        sentences, labels = self.dm_cnn.load_data_and_labels()
        sentences_padded = self.dm_cnn.pad_sentences(sentences)

        x = sentences_padded
        y = np.array(labels)

        y = y.argmax(axis=1)
        # Splitting and shuffling into training and validation set
        self.x_train,self.x_test,self.y_train,self.y_test = self.dm_cnn.split_and_shuffle(x,y,ratio=self.FLAGS.ratio,seed=self.FLAGS.seed)

        print("x_train shape:", len(self.x_train))
        print("x_test shape:", len(self.x_test))
        print("y_train shape:", len(self.y_train))
        print("y_test shape:", len(self.y_test))

    def run_model(self):
        """
        Setting the parameters of the model and then running it.
        """
        print("Running model")

        # Dropout values
        dropout_prob = (0.1, 0.3)
        # The filter size, the number of filters here will correspond to the number of convolution
        # blocks we will have.
        filter_sizes = [3]

        # Input shape of the network depending on the length of the tweet
        # and the embedding word dimension
        input_shape = (self.FLAGS.max_length, self.FLAGS.word_dimension)
        model_input = Input(shape=input_shape)

        # Creating the model
        z = model_input
        # Spatial dropout applied on the input.
        z = SpatialDropout1D(dropout_prob[0])(z)

        # Creating of the convolutional blocks that are corresponding to
        # the length of the filter sizes array.
        conv_blocks = []
        for sz in filter_sizes:
            # Convolution layer using valid padding and relu activation function and stride 1
            conv = Conv1D(filters=self.FLAGS.num_filters,
                                 kernel_size=sz,
                                 padding="valid",
                                 activation="relu",
                                 strides=1)(z)
            # Max Pooling layer
            conv = MaxPooling1D(pool_size=2)(conv)
            # LSTM layer
            conv = LSTM(128)(conv)
            #conv = Flatten()(conv)

            # Appending this structure to the block
            conv_blocks.append(conv)

        # Concatenating all the blocks to form the model.
        z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
        # Dropout layer
        z = Dropout(dropout_prob[1])(z)
        # Fully connected layer
        z = Dense(self.FLAGS.hidden_size, activation="relu")(z)
        # Output layer
        model_output = Dense(1, activation="sigmoid")(z)

        # Creating the model
        model = Model(model_input, model_output)
        # Compiling the model
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

        # Creating checkpoints for the best validation accuracy and drawing graphs.
        filepath="best_weights.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

        callbacks_list = [checkpoint, tbCallBack]

        # Running the model.
        model.fit_generator(self.generator(),steps_per_epoch=self.FLAGS.steps_per_epoch, epochs=self.FLAGS.num_epochs,validation_data= self.generator_validator(),validation_steps=self.FLAGS.validation_step, verbose=2,callbacks=callbacks_list)
        model.save('my_test_model.h5')
