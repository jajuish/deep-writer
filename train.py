import tensorflow as tf
import numpy as np

import argparse
import time
import os
from six.moves import cPickle

from utils import TextLoader
from model import Model

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir",default="data/",
                        help="Mention the directory your input files are on")
    parser.add_argument('--save_dir', type=str, default='save',
                       help='directory to store checkpointed models')
    parser.add_argument("--rnn_size",default=128,type=int,
                        help="Mention the rnn hidden state size")
    parser.add_argument("--batch_size",default=50,type=int,
                        help="Mention the batch size you want here")
    parser.add_argument("--num_layers",default=50,type=int,
                        help="Mention the number of layers you want here")
    parser.add_argument('--seq_length', type=int, default=25,
                        help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                        help='learning rate')

    args = parser.parse_args()

    train(args)

def train(args):
    data_loader = TextLoader(args.data_dir,args.batch_size,args.seq_length)
    args.vocab_size = data_loader.vocab_size

    with open(os.path.join(args.save_dir,'configure.pkl'),'wb') as f:
        cPickle.dump(args,f)
    with open(os.path.join(args.save_dir, 'words_vocab.pkl'), 'wb') as f:
        cPickle.dump((data_loader.words, data_loader.vocab), f)

    model = Model(args)
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())

        for e in range(args.num_epochs):
            #sess.run(tf.assign(model.lr,args.learning_rate * (args.decay_rate ** e)))
            data_loader.reset_batch_pointer()
            state = sess.run(model.initial_state)
            speed = 0
            assign_op = model.epoch_pointer.assign(e)
            sess.run(assign_op)

            for b in range(data_loader.num_batches):
                start = time.time()
                x, y = data_loader.next_batch()
                feed = {model.input_data: x, model.targets: y, model.initial_state: state,
                        model.batch_time: speed}
                summary, train_loss, state, _, _ = sess.run([merged, model.cost, model.final_state,
                                                             model.train_op, model.inc_batch_pointer_op], feed)
                #train_writer.add_summary(summary, e * data_loader.num_batches + b)
                speed = time.time() - start
                if (e * data_loader.num_batches + b) % args.batch_size == 0:
                    print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                        .format(e * data_loader.num_batches + b,
                                args.num_epochs * data_loader.num_batches,
                                e, train_loss, speed))
                '''if (e * data_loader.num_batches + b) % args.save_every == 0 \
                        or (e==args.num_epochs-1 and b == data_loader.num_batches-1): # save for the last result
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step = e * data_loader.num_batches + b)
                    print("model saved to {}".format(checkpoint_path))'''
        #train_writer.close()

if __name__ == '__main__':
    main()
