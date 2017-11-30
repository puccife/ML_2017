import dnc.dnc

class DNCTrainer:

    input_sequence = None
    FLAGS = None


    def __init__(self, input_sequence, FLAGS):
        self.input_sequence = input_sequence
        self.FLAGS = FLAGS


    def run_model(self):
        print("Running model")
          """Runs model on input sequence."""
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
          #Creo la cella dnc
          dnc_core = dnc.DNC(access_config, controller_config, self.FLAGS.num_classes, clip_value)
          initial_state = dnc_core.initial_state(FLAGS.batch_size)
          #Funzione che ritorna una coppia (output,state), dove output in questo caso sara un tensore
          #di forma  [batch_size,max_time,cell.output_size] perche il flag time_major e impostato a False
          #se lo si setta a True invece la forma dell'output diventa [max_time,batch_size,cell.output_size].
          output_sequence, _ = tf.nn.dynamic_rnn(
              cell=dnc_core,
              inputs=self.input_sequence,
              time_major=False,
              initial_state=initial_state)

          return output_sequence


    def train_model(self):
        print("Training model")
        max_lenght = FLAGS.max_lenght
          #Placeholder che andra' a contenere il batch di label relativa alle recensioni
        y_= tf.placeholder(tf.float32,shape=[FLAGS.batch_size,max_lenght,FLAGS.num_classes])

        #Placeholder che andra' a contenere il batch di recensioni opportunamente codificate
        x = tf.placeholder(tf.float32, [FLAGS.batch_size, max_lenght, FLAGS.word_dimension])

        #Placeholder che andra' a contenere il batch di maschere da applicare per il calcolo della cross-entropy
        mask = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,max_lenght])

        #Richiamando il metodo run_model ottengo la sequenza prodotta dalla rete
        output_logits = run_model(x, FLAGS.num_classes)

        #Calcolo della cross entropy totale tra le labels e gli output prodotti dalla rete
        cross = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output_logits)

        #Calcolo l'errore relativo ai singoli batch, applicando una maschera che considera gli output prodotti dalla rete
        #solo negli unrolling corrispondenti a delle parole della recensione e non considerando quelli prodotti in corrispondenza
        #di padding. La maschera applicata fornisce peso via via crescente mano a mano che si procede verso le ultime parole
        #della recensione.
        batch_error = tf.reduce_sum(cross * mask, 1)

        #Faccio la media degli errori dei singoli batch
        total_error = tf.reduce_mean(batch_error)

        #Ricavo la polarita' che la rete ha indicato all'ultima parola di ogni recensione,
        prediction = tf.arg_max(output_logits[:,max_lenght-1], 1)

        #Ricavo la polarita' indicata dalle label
        expected = tf.arg_max(y_[:,max_lenght-1], 1)

        #Ricavo quante predizioni della polarita' sono state fatte correttamente
        correct_prediction = tf.equal(prediction, expected)

        #Ricavo cosi' l'accuratezza ottenuta in questo batch
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        #Set up optimizer with global norm clipping.
        trainable_variables = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(total_error, trainable_variables), FLAGS.max_grad_norm)

        global_step = tf.get_variable(
            name="global_step",
            shape=[],
            dtype=tf.int64,
            initializer=tf.random_uniform_initializer(-1, 1),
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

        #Placeholder che conterra' il learning rate
        learning_rate = tf.placeholder(tf.float32)

        optimizer = tf.train.RMSPropOptimizer(
            learning_rate, epsilon=FLAGS.optimizer_epsilon)

        #Passo di addestramento da eseguire per addestrare la rete
        train_step = optimizer.apply_gradients(
            zip(grads, trainable_variables), global_step=global_step)

        #Oggetto per salvare lo stato della rete.
        saver = tf.train.Saver()



    def __training_step(self):
        print("Training step")
