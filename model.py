import tensorflow as tf
from tensorflow.contrib import rnn, legacy_seq2seq
import random
import numpy as np

#from beam import BeamSearch

class Model():
    def __init__(self,args,infer=False):
        self.args = args

        if infer:
            self.batch_size = 1
            self.seq_length = 1

        cell_fn = rnn.BasicRNNCell
        cells = []
        for _ in range(args.num_layers):
            cell = cell_fn(args.rnn_size)
            cells.append(cell)

        self.cell = cell = rnn.MultiRNNCell(cells) #stacked cells

        # TODO: why used int and not anything else?
        self.input_data = tf.placeholder(tf.int32,[args.batch_size,args.seq_length])
        self.targets = tf.placeholder(tf.int32,[args.batch_size,args.seq_length])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)
        self.batch_pointer = tf.Variable(0, name="batch_pointer", trainable=False, dtype=tf.int32)
        self.inc_batch_pointer_op = tf.assign(self.batch_pointer, self.batch_pointer + 1)
        self.epoch_pointer = tf.Variable(0, name="epoch_pointer", trainable=False)
        self.batch_time = tf.Variable(0.0, name="batch_time", trainable=False)
        tf.summary.scalar("time_batch", self.batch_time)

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w", [args.rnn_size, args.vocab_size])
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size])
            #resizing
            with tf.device("/cpu:0"):
                embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
                inputs = tf.split(tf.nn.embedding_lookup(embedding, self.input_data), args.seq_length, 1)
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        def loop(prev,_):
            prev = tf.matmul(prev,softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev,1))
            return tf.nn.embedding_lookup(embedding,prev_symbol)

        outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if infer else None, scope='rnnlm')
        # TODO: understand this
        output = tf.reshape(tf.concat(outputs, 1), [-1, args.rnn_size])
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        loss = legacy_seq2seq.sequence_loss_by_example([self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([args.batch_size * args.seq_length])],
                args.vocab_size)
        self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        tf.summary.scalar("cost", self.cost)
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),5)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    #def test(self,)
