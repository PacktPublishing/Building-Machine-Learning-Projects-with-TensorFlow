from __future__ import print_function
import numpy as np
import tensorflow as tf

import time
import os
from six.moves import cPickle

from utils import TextLoader
from model import Model

from six import text_type

class arguments: #Generate the arguments class
    save_dir= 'save'
    n=1000
    prime='x:1\n'
    sample=1
    
def main():
    args = arguments()    
    sample(args)   #Pass the argument object

def sample(args):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f) #Load the config from the standard file
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f) #Load the vocabulary
    model = Model(saved_args, True) #Rebuild the model
    with tf.Session() as sess:
        tf.initialize_all_variables().run()  
        saver = tf.train.Saver(tf.all_variables())    
        ckpt = tf.train.get_checkpoint_state(args.save_dir) #Retrieve the chkpoint
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path) #Restore the model
            print(model.sample(sess, chars, vocab, args.n, args.prime, args.sample))
            #Execute the model, generating a n char sequence
            #starting with the prime sequence
if __name__ == '__main__':
    main()
