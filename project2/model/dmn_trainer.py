import time
import os

import pickle 

import numpy as np

import tensorflow as tf

from preprocessing.manipulator import DatasetManipulator
from word_embedding_model.pretrained_glove import GloveTrainer

from dmn.dmn_plus import Config, DMN_PLUS

config = Config()

class DMNTrainer:

    FLAGS = None
    training_generator = None
    testing_generator = None
    dm = None
    training_set = None
    testing_set = None
    missing_voc = None
    word_embeddings = None
    vocab = {}
    ivocab = {}

    def __init__(self, FLAGS):
        """
        Initializing dmn trainer with loaded tf.FLAGS
        :param FLAGS: FLAGS for training
        """
        self.FLAGS = FLAGS


    def load_babi(self):
        """
        load dataset in babi format.
        :return: the preprocessed dataset.
        """
        gt = GloveTrainer(vector_size=config.embed_size, glove_dir=self.FLAGS.glove_dir)
        self.word_embeddings = gt.generate_word_embeddings()
        self.dm = DatasetManipulator(self.FLAGS.dataset_pos,self.FLAGS.dataset_neg, self.FLAGS.dataset_test)
        self.training_set = self.dm.generate_dataset(total_samples=config.num_examples)
        self.testing_set = self.dm.generate_testing_dataset()
        training = self.dm.format_like_babi(self.training_set)
        testing = self.dm.format_like_babi(self.testing_set)
        self.dm.save_reviews_splitted(training, testing)
        train_raw = self.dm.init_babi('./data/tweet_train.txt')
        test_raw = self.dm.init_babi('./data/tweet_test.txt')
        gt.process_word(word_embeddings=self.word_embeddings, word="<eos>", vocab=self.vocab,
                        ivocab=self.ivocab, word_size=config.embed_size, to_return="index")
        train_data = gt.process_input(train_raw.copy(), config.floatX, self.word_embeddings, self.vocab, self.ivocab, config.embed_size, False)
        test_data = gt.process_input(test_raw.copy(), config.floatX, self.word_embeddings, self.vocab, self.ivocab, config.embed_size, False)
        word_embedding = gt.create_embedding(self.word_embeddings, self.ivocab, config.embed_size)
        save_embedding(word_embedding)
        inputs, questions, answers, input_masks, rel_labels = train_data if config.train_mode else test_data
        input_lens, sen_lens, max_sen_len = gt.get_sentence_lens(inputs)
        if config.fine_tuning_mode and config.y_info is not None:
            max_mask_len = config.y_info[5]
        elif config.cross_domain_test and config.y_info is not None:
            max_mask_len = config.y_info[4]
        else:
            max_mask_len = max_sen_len
        q_lens = gt.get_lens(questions)
        print("The max sequence lenght found is " + str(np.max(input_lens)))
        max_q_len = np.max(q_lens)
        max_input_len = min(np.max(input_lens), config.max_allowed_inputs)

        inputs = gt.pad_inputs(inputs, input_lens, max_input_len, "split_sentences", sen_lens, max_sen_len)
        input_masks = np.zeros(len(inputs))

        questions = gt.pad_inputs(questions, q_lens, max_q_len)
        answers = np.stack(answers)

        rel_labels = np.zeros((len(rel_labels), len(rel_labels[0])))

        for i, tt in enumerate(rel_labels):
            rel_labels[i] = np.array(tt, dtype=int)

        if config.train_mode:
            reviews_train_n = int(config.num_examples*(config.training_ratio))
            train = questions[:reviews_train_n], inputs[:reviews_train_n], q_lens[:reviews_train_n], \
                input_lens[:reviews_train_n], input_masks[:reviews_train_n], answers[:reviews_train_n], \
                rel_labels[:reviews_train_n]
            valid = questions[reviews_train_n:], inputs[reviews_train_n:], q_lens[reviews_train_n:], \
                    input_lens[reviews_train_n:], input_masks[reviews_train_n:], answers[reviews_train_n:], \
                    rel_labels[reviews_train_n:]
            print("Training on: " + str(len(train[0])))
            print("Validating on: " + str(len(valid[0])))
            return train, valid, word_embedding, max_q_len, max_input_len, max_mask_len, rel_labels.shape[1], len(self.vocab)
        else:
            test = questions, inputs, q_lens, input_lens, input_masks, answers, rel_labels
            return test, word_embedding, max_q_len, max_input_len, max_mask_len, rel_labels.shape[1], len(self.vocab)

    def train_DMN(self):
        """
        Function used to train DMN network
        """
        model = DMN_PLUS(config, self)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        # TensorFlow GPU options
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        #tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9
        best_overall_val_loss = float('inf')
        with tf.Session(config=tf_config) as session:
            sum_dir = 'summaries/train/' + time.strftime("%Y-%m-%d %H %M")
            if not os.path.exists(sum_dir):
                os.makedirs(sum_dir)
            train_writer = tf.summary.FileWriter(sum_dir, session.graph)
            session.run(init)
            best_val_epoch = 0
            prev_epoch_loss = float('inf')
            best_val_loss = float('inf')
            best_val_accuracy = 0.0
            print('==> starting training')
            start = time.time()
            for epoch in range(config.max_epochs):
                print('Epoch {}'.format(epoch))

                print(len(model.train[0]))
                print(len(model.valid[0]))
                
                if(len(model.train[0]) != 0):
                    train_loss, train_accuracy = model.run_epoch(
                        session, model.train, epoch, train_writer,
                        train_op=model.train_step, train=True)
                    print('Training accuracy: {}'.format(train_accuracy))
                    print('\nTraining loss: {}'.format(train_loss))
                    # anneal
                    if train_loss > prev_epoch_loss * model.config.anneal_threshold:
                        model.config.lr /= model.config.anneal_by
                    print('annealed lr to %f' % model.config.lr)
                    prev_epoch_loss = train_loss
                    if epoch - best_val_epoch > config.early_stopping:
                        break
                if(len(model.valid[0]) != 0):
                    valid_loss, valid_accuracy = model.run_epoch(session, model.valid)
                    print('Validation loss: {}'.format(valid_loss))
                    print('Vaildation accuracy: {}'.format(valid_accuracy))
                    if valid_loss < best_val_loss:
                        best_val_loss = valid_loss
                        best_val_epoch = epoch
                        if best_val_loss < best_overall_val_loss:
                            print('Saving weights')
                            best_overall_val_loss = best_val_loss
                            best_val_accuracy = valid_accuracy
                            saver.save(session, 'weights/task' + str(model.config.babi_id) +'.weights')
                
            print('Total time: {}'.format(time.time() - start))

            print('Best validation accuracy:', best_val_accuracy)

        if config.clean_datasets:
            print('==> Cleaning splitted datasets')
            data_input.delete_revs_file(config.babi_id, config.babi_test_id)
            if args.fine_tuning:
                data_input.delete_revs_file(args.destination_task, args.destination_task)

def load_embedding():
    """
    Function used to load embedding word vector
    :return: loaded embedding word vector
    """
    with open('./embedding.pkl', 'rb') as f:
        return pickle.load(f)

def save_embedding(obj):
    """
    Function used to save the word embedding
    :param obj: word embedding dictionary
    """
    with open('./embedding.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
