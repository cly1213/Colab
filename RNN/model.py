import tensorflow as tf


class RNN():
    def __init__(self, batch_size, sequence_len, num_dim):
        self.batch_size = batch_size
        self.seq_len = sequence_len
        self.lstm_size = 16
        self.num_layers = 2
        self.num_dim = num_dim
        #write your code here
        self.inputs = tf.placeholder(tf.float32, shape = [batch_size, sequence_len, num_dim], name = 'inputs')
    
        self.labels = tf.placeholder(tf.float32, shape = [batch_size, num_dim], name='labels')
        
        self.lr = tf.placeholder(tf.float32, shape = [], name = "learning_rate")
    
        self.is_training = tf.placeholder(tf.bool, name = 'is_training')
    
        with tf.variable_scope('lstm_layers'):
            self.build_lstm()
        
        with tf.variable_scope('output_layers'):
            self.build_output()
    

    def build_lstm(self):
    #write your code here
        cell = tf.nn.rnn_cell.LSTMCell(self.lstm_size, state_is_tuple=True)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=0.5)
        self.initial_state = cell.zero_state(self.batch_size, tf.float32)
        self.lstm_outputs, self.final_state = tf.nn.dynamic_rnn(cell, self.inputs, dtype=tf.float32, initial_state=self.initial_state)

    def build_output(self):
    #write your code here
        self.lsop = tf.reshape(self.lstm_outputs, [self.batch_size, -1])
        
        self.predictions = tf.layers.dense(inputs = self.lsop, units = 512,)
        
        self.predictions = tf.layers.dense(inputs = self.predictions, units = self.num_dim,)
        
        self.cost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.labels, self.predictions))))
        
        self.train_network = tf.train.AdamOptimizer(self.lr).minimize(self.cost)
