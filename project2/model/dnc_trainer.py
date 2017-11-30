
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


    def __training_step(self):
        print("Training step")
