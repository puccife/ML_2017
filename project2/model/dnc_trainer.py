import time
import json
import math

import tensorflow as tf
import dnc.dnc as dnc

from preprocessing.manipulator import DatasetManipulator
from word_embedding_model.pretrained_glove import GloveTrainer
import pandas as pd

class DNCTrainer:

    FLAGS = None
    training_generator = None
    testing_generator = None
    dm = None
    training_set = None
    validation_set = None
    testing_set = None
    missing_voc = None

    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.__init_model()


    def __init_model(self):
        """
        Function used to initialize DNC trainer
        """
        gt = GloveTrainer(vector_size=self.FLAGS.word_dimension, glove_dir=self.FLAGS.glove_dir)
        word_embeddings = gt.generate_word_embeddings()
        self.dm = DatasetManipulator(self.FLAGS.dataset_pos,self.FLAGS.dataset_neg, self.FLAGS.dataset_test)
        tweets = self.dm.generate_dataset(total_samples=self.FLAGS.total_samples)
        test = self.dm.generate_testing_dataset()
        tweets_glove = gt.manipulate_dataset(tweets.copy(), word_embeddings)
        self.training_set, self.validation_set = self.dm.split_and_shuffle(tweets_glove, ratio=self.FLAGS.ratio, seed=self.FLAGS.seed)
        self.testing_set = gt.manipulate_dataset(test.copy(), word_embeddings)
        self.__create_generators()
        print("Training on: " + str(len(self.training_set)) + " elements")

    def __create_generators(self):
        """
        Function used to create the generators for training and testing
        """
        self.training_generator = self.dm.get_generator(self.training_set, self.FLAGS)
        self.testing_generator = self.dm.get_generator(self.testing_set, self.FLAGS)

    def run_model(self, input_sequence):
        """
        Function that creates the structure of the DNC and returns the expected output
        :param input_sequence: the input sequence of the DNC
        :return: expected output
        """
        print("Running model")
        access_config = {
          "memory_size": self.FLAGS.memory_size,
          "word_size": self.FLAGS.word_size,
          "num_reads": self.FLAGS.num_read_heads,
          "num_writes": self.FLAGS.num_write_heads,
        }
        controller_config = {
          "hidden_size": self.FLAGS.hidden_size,
        }

        clip_value = self.FLAGS.clip_value
        #Creating dnc cell
        dnc_core = dnc.DNC(access_config, controller_config, self.FLAGS.num_classes, clip_value)
        initial_state = dnc_core.initial_state(self.FLAGS.batch_size)

        # This function returns a couple (output, state) where output in this case will be a tensor
        # of shape [batch_size, max_time, cell.output_size] since flag time_major is set to false
        output_sequence, _ = tf.nn.dynamic_rnn(
          cell=dnc_core,
          inputs=input_sequence,
          time_major=False,
          initial_state=initial_state)

        return output_sequence


    def train_model(self):
        print("Training model")
        max_lenght = self.FLAGS.max_lenght
          #Placeholder for the label
        y_= tf.placeholder(tf.float32,shape=[self.FLAGS.batch_size,max_lenght,self.FLAGS.num_classes])

        #Placeholder for the input sequence
        x = tf.placeholder(tf.float32, [self.FLAGS.batch_size, max_lenght, self.FLAGS.word_dimension])

        #Placeholder for the mask -> cross-entropy
        mask = tf.placeholder(tf.float32,shape=[self.FLAGS.batch_size,max_lenght])

        #Getting the expected output from the network
        output_logits = self.run_model(x)

        #Function to get the total cross-entropy loss
        cross = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output_logits)

        # Getting the relative error for each batch, applying mask that applies weights to the position of each word.
        batch_error = tf.reduce_sum(cross * mask, 1)

        #Avg of single batch error
        total_error = tf.reduce_mean(batch_error)

        #Getting prediction of the network
        prediction = tf.argmax(output_logits[:,max_lenght-1], 1)

        #Getting expected label
        expected = tf.argmax(y_[:,max_lenght-1], 1)

        #Getting number of correct prediction
        correct_prediction = tf.equal(prediction, expected)

        #Getting accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        #Set up optimizer with global norm clipping.
        trainable_variables = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(total_error, trainable_variables), self.FLAGS.max_grad_norm)

        global_step = tf.get_variable(
            name="global_step",
            shape=[],
            dtype=tf.int64,
            initializer=tf.random_uniform_initializer(-1, 1),
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

        #Placeholder for learning rate
        learning_rate = tf.placeholder(tf.float32)

        optimizer = tf.train.RMSPropOptimizer(
            learning_rate, epsilon=self.FLAGS.optimizer_epsilon)

        #Training step
        train_step = optimizer.apply_gradients(
            zip(grads, trainable_variables), global_step=global_step)

        #Network saver
        saver = tf.train.Saver()

        #Parameters to save the network
        if self.FLAGS.checkpoint_interval > 0:
          hooks = [
              tf.train.CheckpointSaverHook(
                  checkpoint_dir=self.FLAGS.checkpoint_dir,
                  save_steps=self.FLAGS.checkpoint_interval,
                  saver=saver)
          ]
        else:
          hooks = []

        #Log file writer
        date = time.strftime("%b%d%H:%M:%S")
        outputFile = open("Experiment"+date+".txt",'w')

        with open("config/configuration.json") as data_file:
          configuration = json.load(data_file)
          outputFile.write(json.dumps(configuration))

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.4

        with tf.train.SingularMonitoredSession(
            hooks=hooks, checkpoint_dir=self.FLAGS.checkpoint_dir,config=config) as sess:

          # Getting first batch to train the network
          datasetTrain = next(self.training_generator)

          #Variable to set the initializing parameter
          glob = True
          #Initializing
          start_iteration = sess.run(global_step,{x:(datasetTrain[0]),
                                                  y_:(datasetTrain[1]),
                                                  mask:(datasetTrain[2]),
                                                  learning_rate: self.FLAGS.learning_rate})

          #Learning rate decreasing.
          if self.FLAGS.num_epochs > 10:
              delta = (self.FLAGS.learning_rate-self.FLAGS.final_learning_rate)/9
          else:
              delta = (self.FLAGS.learning_rate-self.FLAGS.final_learning_rate)/(self.FLAGS.num_epochs-1)

          #Saving best training accuracy
          best_train_accuracy  = 0

          for epochs in range(self.FLAGS.num_epochs):
              if not glob:
                  self.__create_generators()
              if self.FLAGS.num_epochs > 1 :
                  start_iteration = 0

              total_accuracy = 0
              total_entropy = 0
              train_accuracy = 0
              newLearningRate = self.FLAGS.learning_rate - delta * (epochs)

              #Stop decreasing lr after 10 epochs
              if epochs > 9:
                  newLearningRate = self.FLAGS.final_learning_rate

              date = time.strftime("%H:%M:%S")
              # tf.logging.info("Memory usage %f Mb",memory_usage()['rss']/1000)
              tf.logging.info("Ora: %s Epoca %d, Learning rate: %f\n",date,epochs,newLearningRate)
              info1 = '\nOra: '+date+', Epoca '+str(epochs)+ ', Learning rate: '+str(newLearningRate)
              outputFile.write(info1)
              #################################################TRAINING########################################################

              for train_iteration in range(start_iteration, int(self.FLAGS.total_samples*self.FLAGS.ratio//self.FLAGS.batch_size)):
                  if glob:
                      glob = False
                  else:
                      datasetTrain = next(self.training_generator)
                  #Training
                  _, act_accuracy, entropy = sess.run([train_step, accuracy, total_error],
                                                      {x: (datasetTrain[0]),
                                                       y_: (datasetTrain[1]),
                                                       mask: (datasetTrain[2]),
                                                       learning_rate: newLearningRate})
                  #Validity checks
                  val = float(entropy)
                  if math.isnan(val):
                      print('Detected NaN')
                      s=str("Input precedente: \n")
                      outputFile.write(s)
                      outputFile.write(str(datasetTrain[0]))
                      outputFile.write("\n")
                      outputFile.write(str(datasetTrain[1]))
                      outputFile.write("\n")
                      outputFile.write(str(datasetTrain[2]))
                      outputFile.write("\n")
                      outputFile.close()
                      exit(5)

                  total_accuracy += act_accuracy
                  total_entropy += entropy
                  train_accuracy += act_accuracy

                  #Info intervals
                  if (train_iteration + 1) % self.FLAGS.report_interval == 0:
                      date = time.strftime("%H:%M:%S")
                      tf.logging.info("Time: %s ,%d: Avg training accuracy %f.\nAvg Cross entropy: %f\n",
                                      date,train_iteration+1, total_accuracy / self.FLAGS.report_interval, total_entropy / self.FLAGS.report_interval)
                      info2 = "\Time: "+date+" "+str(train_iteration+1)+": Avg training accuracy: "+str(total_accuracy / self.FLAGS.report_interval)+\
                              "\nAvg Cross entropy: "+str(total_entropy / self.FLAGS.report_interval)+"\n"
                      outputFile.write(info2)
                      total_accuracy = 0
                      total_entropy = 0

              #Info interval
              tf.logging.info("\nEpoch: %d,Iteration: %d, Average Training accuracy: %f\n",
                              epochs, train_iteration+1, train_accuracy / (train_iteration+1))

              info3 = "\nEpoch: "+str(epochs)+",Iteration: "+str(train_iteration+1)+", Average Training accuracy: "+str(train_accuracy / (train_iteration+1))+"\n"
              outputFile.write(info3)

              epoch_train_accuracy = train_accuracy / (train_iteration+1)

              #Saving best result
              if epoch_train_accuracy> best_train_accuracy:
                  best_train_accuracy = epoch_train_accuracy

          predictions = []
          for test_iteration in range(0,(self.FLAGS.num_testing_iterations//self.FLAGS.batch_size)):
              datasetTest = next(self.testing_generator)
              res_prediction = sess.run(prediction,
                                      {x: datasetTest[0],
                                       y_: datasetTest[1],
                                       mask: datasetTest[2]})

              predictions.extend(res_prediction)
          print("How many predictions? " + str(len(predictions)))
          final_predictions = pd.DataFrame(predictions.copy())
          final_predictions.index += 1
          final_predictions = final_predictions.reset_index()
          final_predictions.columns = ['Id', 'Prediction']
          final_predictions = final_predictions.set_index('Id')
          final_predictions[final_predictions['Prediction'] == 0] = -1
          final_predictions.to_csv('./predictions_csv/DNC_prediction.csv')
          outputFile.close()

    def test_model(self):
        """
        Function to test the model 
        """
        print("Testing model")
        max_lenght = self.FLAGS.max_lenght

        y_= tf.placeholder(tf.float32,shape=[self.FLAGS.batch_size,max_lenght,self.FLAGS.num_classes])

        x = tf.placeholder(tf.float32, [self.FLAGS.batch_size, max_lenght, self.FLAGS.word_dimension])

        mask = tf.placeholder(tf.float32,shape=[self.FLAGS.batch_size,max_lenght])

        output_logits = self.run_model(x)

        cross = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output_logits)

        batch_error = tf.reduce_sum(cross * mask, 1)

        total_error = tf.reduce_mean(batch_error)

        prediction = tf.argmax(output_logits[:,max_lenght-1], 1)

        expected = tf.argmax(y_[:,max_lenght-1], 1)

        correct_prediction = tf.equal(prediction, expected)

        #Set up optimizer with global norm clipping.
        trainable_variables = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(total_error, trainable_variables), self.FLAGS.max_grad_norm)

        global_step = tf.get_variable(
            name="global_step",
            shape=[],
            dtype=tf.int64,
            initializer=tf.random_uniform_initializer(-1, 1),
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

        #Placeholder for lr
        learning_rate = tf.placeholder(tf.float32)

        optimizer = tf.train.RMSPropOptimizer(
            learning_rate, epsilon=self.FLAGS.optimizer_epsilon)

        #State saver
        saver = tf.train.Saver()

        #Saver infos
        if self.FLAGS.checkpoint_interval > 0:
          hooks = [
              tf.train.CheckpointSaverHook(
                  checkpoint_dir=self.FLAGS.checkpoint_dir,
                  save_steps=self.FLAGS.checkpoint_interval,
                  saver=saver)
          ]
        else:
          hooks = []

        #Log files.
        date = time.strftime("%b%d%H:%M:%S")
        outputFile = open("Experiment"+date+".txt",'w')

        with open("config/configuration.json") as data_file:
          configuration = json.load(data_file)
          outputFile.write(json.dumps(configuration))

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.4

        # Predicting
        with tf.train.SingularMonitoredSession(
            hooks=hooks, checkpoint_dir=self.FLAGS.checkpoint_dir,config=config) as sess:
          predictions = []
          for test_iteration in range(0,(self.FLAGS.num_testing_iterations//self.FLAGS.batch_size)):
              datasetTest = next(self.testing_generator)
              res_prediction = sess.run(prediction,
                                      {x: datasetTest[0],
                                       y_: datasetTest[1],
                                       mask: datasetTest[2]})

              predictions.extend(res_prediction)
          print("How many predictions? " + str(len(predictions)))
          final_predictions = pd.DataFrame(predictions.copy())
          final_predictions.index += 1
          final_predictions = final_predictions.reset_index()
          final_predictions.columns = ['Id', 'Prediction']
          final_predictions = final_predictions.set_index('Id')
          final_predictions[final_predictions['Prediction'] == 0] = -1
          final_predictions.to_csv('./predictions_csv/DNC_prediction.csv')
          outputFile.close()