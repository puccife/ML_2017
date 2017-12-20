from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import tensorflow as tf

from dmn.attention_gru_cell import AttentionGRUCell

class Config(object):
    """Holds model hyperparams and data information."""
    num_examples = 2000000
    embed_size = 200
    num_classes = 2
    batch_size = 50
    hidden_size = 128
    max_epochs = 5
    early_stopping = 3
    dropout = 0.9
    lr = 0.001
    l2 = 0.001
    cap_grads = False
    max_grad_val = 15
    noisy_grads = False

    # Initialize word embeddings with GloVe or Google vectors, can be "google" or "glove"
    word2vec_init = None
    embedding_init = np.sqrt(3)

    # set to zero with strong supervision to only train gates
    strong_supervision = False
    beta = 1

    # NOTE not currently used hence non-sensical anneal_threshold
    anneal_threshold = 1000
    anneal_by = 1.5

    num_hops = 3
    num_attention_features = 6

    max_allowed_inputs = 23
    # Added in order to sample different parts of the dataset
    sampling_offset_factor = 1
    # If not None, it is used to balance the polarity of training samples in 2 classes case
    negative_samples_rate = None
    # if set, it's used to decide testing set polarity in binary classification
    negative_test_samples_rate = None

    floatX = np.float32

    seed = 33

    babi_id = "DMN1"
    babi_test_id = ""

    # Added to split datasets in sentiment classification
    training_ratio = 0.9
    validation_ratio = 0.1

    # To improve
    clean_datasets = False

    # train_mode = False
    train_mode = False
    
    # Transfer Learning parameters
    fine_tuning_mode = False
    cross_domain_test = False
    y_info = None

    # Use generators to stock less memory while running, beta... don't use
    save_memory = False


def _add_gradient_noise(t, stddev=1e-3, name=None):
    """Adds gradient noise as described in http://arxiv.org/abs/1511.06807
    The input Tensor `t` should be a gradient.
    The output will be `t` + gaussian noise.
    0.001 was said to be a good fixed value for memory networks."""
    with tf.op_scope([t, stddev], name, "add_gradient_noise") as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)


# from https://github.com/domluna/memn2n
def _position_encoding(sentence_size, embedding_size):
    """Position encoding described in section 4.1 in "End to End Memory Networks" (http://arxiv.org/pdf/1503.08895v5.pdf)"""
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size + 1
    le = embedding_size + 1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i - 1, j - 1] = (i - (le - 1) / 2) * (j - (ls - 1) / 2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    return np.transpose(encoding)


# Generators are still to implement...
def create_qp_generator(qp, total_steps, batch_size):
    for step in range(total_steps):
        index = range(step * batch_size, (step + 1) * batch_size)
        yield qp[index]


def create_ip_generator(ip, total_steps, batch_size):
    for step in range(total_steps):
        index = range(step * batch_size, (step + 1) * batch_size)
        yield ip[index]


def create_ql_generator(ql, total_steps, batch_size):
    for step in range(total_steps):
        index = range(step * batch_size, (step + 1) * batch_size)
        yield ql[index]


def create_il_generator(il, total_steps, batch_size):
    for step in range(total_steps):
        index = range(step * batch_size, (step + 1) * batch_size)
        yield il[index]


def create_a_generator(a, total_steps, batch_size):
    for step in range(total_steps):
        index = range(step * batch_size, (step + 1) * batch_size)
        yield a[index]


def create_r_generator(r, total_steps, batch_size):
    for step in range(total_steps):
        index = range(step * batch_size, (step + 1) * batch_size)
        yield r[index]


def create_data_generator(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


class DMN_PLUS(object):
    def load_data(self, debug=False):
        """Loads train/valid/test data and sentence encoding"""
        if self.config.train_mode:
            self.train, self.valid, self.word_embedding, self.max_q_len, self.max_input_len, self.max_sen_len, self.num_supporting_facts, self.vocab_size = self.dmn_trainer.load_babi()
        else:
            self.test, self.word_embedding, self.max_q_len, self.max_input_len, self.max_sen_len, self.num_supporting_facts, self.vocab_size = self.dmn_trainer.load_babi()
        self.encoding = _position_encoding(self.max_sen_len, self.config.embed_size)

    def add_placeholders(self):
        """add data placeholder to graph"""
        self.question_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size, self.max_q_len))
        self.input_placeholder = tf.placeholder(tf.int32,
                                                shape=(self.config.batch_size, self.max_input_len, self.max_sen_len))

        self.question_len_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size,))
        self.input_len_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size,))

        self.answer_placeholder = tf.placeholder(tf.int64, shape=(self.config.batch_size,))

        self.rel_label_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size, self.num_supporting_facts))

        self.dropout_placeholder = tf.placeholder(tf.float32)

    def get_predictions(self, output):
        preds = tf.nn.softmax(output)
        pred = tf.argmax(preds, 1)
        return pred

    def add_loss_op(self, output):
        """Calculate loss"""
        # optional strong supervision of attention with supporting facts
        gate_loss = 0
        if self.config.strong_supervision:
            for i, att in enumerate(self.attentions):
                labels = tf.gather(tf.transpose(self.rel_label_placeholder), 0)
                gate_loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=att, labels=labels))

        loss = self.config.beta * tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=self.answer_placeholder)) + gate_loss

        # add l2 regularization for all variables except biases
        for v in tf.trainable_variables():
            if not 'bias' in v.name.lower():
                loss += self.config.l2 * tf.nn.l2_loss(v)

        tf.summary.scalar('loss', loss)

        return loss

    def add_training_op(self, loss):
        """Calculate and apply gradients"""
        opt = tf.train.AdamOptimizer(learning_rate=self.config.lr)
        gvs = opt.compute_gradients(loss)

        # optionally cap and noise gradients to regularize
        if self.config.cap_grads:
            gvs = [(tf.clip_by_norm(grad, self.config.max_grad_val), var) for grad, var in gvs]
        if self.config.noisy_grads:
            gvs = [(_add_gradient_noise(grad), var) for grad, var in gvs]

        train_op = opt.apply_gradients(gvs)
        return train_op

    def get_question_representation(self, embeddings):
        """Get question vectors via embedding and GRU"""
        questions = tf.nn.embedding_lookup(embeddings, self.question_placeholder)

        gru_cell = tf.contrib.rnn.GRUCell(self.config.hidden_size)
        _, q_vec = tf.nn.dynamic_rnn(gru_cell,
                                     questions,
                                     dtype=np.float32,
                                     sequence_length=self.question_len_placeholder
                                     )

        return q_vec

    def get_input_representation(self, embeddings):
        """Get fact (sentence) vectors via embedding, positional encoding and bi-directional GRU"""
        # get word vectors from embedding
        inputs = tf.nn.embedding_lookup(embeddings, self.input_placeholder)

        # use encoding to get sentence representation
        inputs = tf.reduce_sum(inputs * self.encoding, 2)

        forward_gru_cell = tf.contrib.rnn.GRUCell(self.config.hidden_size)
        backward_gru_cell = tf.contrib.rnn.GRUCell(self.config.hidden_size)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            forward_gru_cell,
            backward_gru_cell,
            inputs,
            dtype=np.float32,
            sequence_length=self.input_len_placeholder
        )

        # f<-> = f-> + f<-
        fact_vecs = tf.reduce_sum(tf.stack(outputs), axis=0)

        # apply dropout
        fact_vecs = tf.nn.dropout(fact_vecs, self.dropout_placeholder)

        return fact_vecs

    def get_attention(self, q_vec, prev_memory, fact_vec, reuse):
        """Use question vector and previous memory to create scalar attention for current fact"""
        with tf.variable_scope("attention", reuse=reuse):
            features = [fact_vec * q_vec,
                        fact_vec * prev_memory,
                        tf.abs(fact_vec - q_vec),
                        tf.abs(fact_vec - prev_memory)]

            feature_vec = tf.concat(features, 1)

            attention = tf.contrib.layers.fully_connected(feature_vec,
                                                          self.config.embed_size,
                                                          activation_fn=tf.nn.tanh,
                                                          reuse=reuse, scope="fc1")

            attention = tf.contrib.layers.fully_connected(attention,
                                                          1,
                                                          activation_fn=None,
                                                          reuse=reuse, scope="fc2")

        return attention

    def generate_episode(self, memory, q_vec, fact_vecs, hop_index):
        """Generate episode by applying attention to current fact vectors through a modified GRU"""

        attentions = [tf.squeeze(
            self.get_attention(q_vec, memory, fv, bool(hop_index) or bool(i)), axis=1)
            for i, fv in enumerate(tf.unstack(fact_vecs, axis=1))]

        attentions = tf.transpose(tf.stack(attentions))
        self.attentions.append(attentions)
        attentions = tf.nn.softmax(attentions)
        attentions = tf.expand_dims(attentions, axis=-1)

        reuse = True if hop_index > 0 else False

        # concatenate fact vectors and attentions for input into attGRU
        gru_inputs = tf.concat([fact_vecs, attentions], 2)

        with tf.variable_scope('attention_gru', reuse=reuse):
            _, episode = tf.nn.dynamic_rnn(AttentionGRUCell(self.config.hidden_size),
                                           gru_inputs,
                                           dtype=np.float32,
                                           sequence_length=self.input_len_placeholder
                                           )

        return episode

    def add_answer_module(self, rnn_output, q_vec):
        """Linear softmax answer module"""

        rnn_output = tf.nn.dropout(rnn_output, self.dropout_placeholder)

        output = tf.layers.dense(tf.concat([rnn_output, q_vec], 1),
                                 self.vocab_size,
                                 activation=None)

        return output

    def inference(self):
        """Performs inference on the DMN model"""

        # set up embedding
        embeddings = tf.Variable(self.word_embedding.astype(np.float32), name="Embedding")

        # input fusion module
        with tf.variable_scope("question", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> get question representation')
            q_vec = self.get_question_representation(embeddings)

        with tf.variable_scope("input", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> get input representation')
            fact_vecs = self.get_input_representation(embeddings)

        # keep track of attentions for possible strong supervision
        self.attentions = []

        # memory module
        with tf.variable_scope("memory", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> build episodic memory')

            # generate n_hops episodes
            prev_memory = q_vec

            for i in range(self.config.num_hops):
                # get a new episode
                print('==> generating episode', i)
                episode = self.generate_episode(prev_memory, q_vec, fact_vecs, i)

                # untied weights for memory update
                with tf.variable_scope("hop_%d" % i):
                    prev_memory = tf.layers.dense(tf.concat([prev_memory, episode, q_vec], 1),
                                                  self.config.hidden_size,
                                                  activation=tf.nn.relu)

            output = prev_memory

        # pass memory module output through linear answer module
        with tf.variable_scope("answer", initializer=tf.contrib.layers.xavier_initializer()):
            output = self.add_answer_module(output, q_vec)

        return output

    def run_epoch(self, session, data, num_epoch=0, train_writer=None, train_op=None, verbose=2, train=False):
        """
        Function used to run one epoch of training or predict from the test set
        :param session: tensorflow session
        :param data: data 
        :param num_epoch: specified in the configuration file
        :param train_writer: log writer for tensorboard
        :param train_op: training operation
        :param verbose: verbose output description
        :param train: training flag - not used - get the flag from the configuration file
        :return: the results of training or testing
        """
        config = self.config
        dp = config.dropout
        if train_op is None:
            train_op = tf.no_op()
            dp = 1

        # debug
        print("Size: " + str(len(data[0])))

        print(config.batch_size)

        total_steps = len(data[0]) // config.batch_size

        # debug
        print("Total steps: " + str(total_steps))
        total_loss = []
        accuracy = 0

        # shuffle data
        qp, ip, ql, il, im, a, r = data

        # Shuffling can cause MemoryError
        #if self.config.train_mode:
            #p = np.random.permutation(len(data[0]))
            #qp = qp[p]
            #ip = ip[p]
            #ql = ql[p]
            #il = il[p]
            #im = im[p]
            #a = a[p]
            #r = r[p]

        predictions = []
        for step in range(total_steps):
            index = range(step * config.batch_size, (step + 1) * config.batch_size)

            feed = {self.question_placeholder: qp[index],
                    self.input_placeholder: ip[index],
                    self.question_len_placeholder: ql[index],
                    self.input_len_placeholder: il[index],
                    self.answer_placeholder: a[index],
                    self.rel_label_placeholder: r[index],
                    self.dropout_placeholder: dp}

            loss, pred, summary, _ = session.run(
                [self.calculate_loss, self.pred, self.merged, train_op], feed_dict=feed)

            if train_writer is not None:
                train_writer.add_summary(summary, num_epoch * total_steps + step)

            answers = a[step * config.batch_size:(step + 1) * config.batch_size]
            accuracy += np.sum(pred == answers) / float(len(answers))

            predictions.extend(pred)
            total_loss.append(loss)
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(
                    step, total_steps, np.mean(total_loss)))
                sys.stdout.flush()

        if verbose:
            sys.stdout.write('\r')

        if total_steps == 0:
            total_steps = 1

        if not self.config.train_mode:
            return predictions
        return np.mean(total_loss), accuracy / float(total_steps)

    def __init__(self, config, dmn_trainer):
        """
        Initializing dmn network
        :param config: configuration file loaded
        :param dmn_trainer: dmn trainer that trains the network.
        """
        self.dmn_trainer = dmn_trainer
        self.config = config
        self.variables_to_save = {}
        self.load_data(debug=False)
        self.add_placeholders()
        self.output = self.inference()
        self.pred = self.get_predictions(self.output)
        self.calculate_loss = self.add_loss_op(self.output)
        self.train_step = self.add_training_op(self.calculate_loss)
        self.merged = tf.summary.merge_all()
