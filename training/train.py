"""
Main trainer function
"""
import theano
import theano.tensor as tensor

import cPickle as pkl
import numpy
import copy

import os
import warnings
import sys
import time
import pdb
import argparse

import homogeneous_data

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from utils import *
from layers import get_layer, param_init_fflayer, fflayer, param_init_gru, gru_layer
from optim import adam
from model import init_params, build_model
from vocab import load_dictionary

# main trainer
def trainer(X, 
            dim_word=300, # word vector dimensionality
            dim=300, # the number of GRU units
            encoder='gru',
            decoder='gru',
            max_epochs=4,
            dispFreq=1,
            decay_c=0.1,
            grad_clip=8.0,
            n_words=64526,
            maxlen_w=30,
            optimizer='adam',
            batch_size=32,
            saveto='/home/shunan/Code/skip-thoughts/experiments/imdb/test_model/test_uni.npz',
            dictionary='/home/shunan/Code/skip-thoughts/experiments/imdb/word_dicts.pkl',
            saveFreq=1000,
            dataset='imdb',
            reload_=False):

    theano.config.allow_gc = True
    # Model options
    model_options = {}
    model_options['dim_word'] = dim_word
    model_options['dim'] = dim
    model_options['encoder'] = encoder
    model_options['decoder'] = decoder 
    model_options['max_epochs'] = max_epochs
    model_options['dispFreq'] = dispFreq
    model_options['decay_c'] = decay_c
    model_options['grad_clip'] = grad_clip
    model_options['n_words'] = n_words
    model_options['maxlen_w'] = maxlen_w
    model_options['optimizer'] = optimizer
    model_options['batch_size'] = batch_size
    model_options['saveto'] = saveto
    model_options['dictionary'] = dictionary
    model_options['saveFreq'] = saveFreq
    model_options['dataset'] = dataset
    model_options['reload_'] = reload_

    print model_options

    # reload options
    if reload_ and os.path.exists(saveto):
        print 'reloading...' + saveto
        with open('%s.pkl'%saveto, 'rb') as f:
            model_options = pkl.load(f)

    # load dictionary
    print 'Loading dictionary...'
    worddict = load_dictionary(dictionary)

    # Inverse dictionary
    word_idict = dict()
    for kk, vv in worddict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    print 'Building model'
    params = init_params(model_options)
    # reload parameters
    if reload_ and os.path.exists(saveto):
        params = load_params(saveto, params)

    tparams = init_tparams(params)

    trng, x, x_mask, y, y_mask, z, z_mask, \
          opt_ret, \
          cost = \
          build_model(tparams, model_options)
    inps = [x, x_mask, y, y_mask, z, z_mask]

    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, cost, profile=False)
    print 'Done'

    # weight decay, if applicable
    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # after any regularizer
    print 'Building f_cost...',
    f_cost = theano.function(inps, cost, profile=False)
    print 'Done'

    print 'Done'
    print 'Building f_grad...',
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    f_grad_norm = theano.function(inps, [(g**2).sum() for g in grads], profile=False)
    f_weight_norm = theano.function([], [(t**2).sum() for k,t in tparams.iteritems()], profile=False)

    if grad_clip > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g**2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (grad_clip**2),
                                           g / tensor.sqrt(g2) * grad_clip,
                                           g))
        grads = new_grads

    lr = tensor.scalar(name='lr')
    print 'Building optimizers...',
    # (compute gradients), (updates parameters)
    f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps, cost)

    print 'Optimization'

    # Each sentence in the minibatch have same length (for encoder)
    # trainX = homogeneous_data.grouper(X)
    if dataset == 'amazon':
        trainX = homogeneous_data.amazon_grouper()
    elif dataset == 'imdb':
        trainX = homogeneous_data.imdb_grouper()
    train_iter = homogeneous_data.HomogeneousData(trainX, batch_size=batch_size, maxlen=maxlen_w)

    uidx = 0
    lrate = 0.01
    for eidx in xrange(max_epochs):
        n_samples = 0

        print 'Epoch ', eidx

        for x, y, z in train_iter:
            n_samples += len(x)
            uidx += 1

            x, x_mask, y, y_mask, z, z_mask = homogeneous_data.prepare_data(x, y, z, worddict, maxlen=maxlen_w, n_words=n_words)

            if x is None:
                print 'Minibatch with zero sample under length ', maxlen_w
                uidx -= 1
                continue

            ud_start = time.time()
            cost = f_grad_shared(x, x_mask, y, y_mask, z, z_mask)
            f_update(lrate)
            ud = time.time() - ud_start

            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'NaN detected'
                return 1., 1., 1.

            if numpy.mod(uidx, dispFreq) == 0:
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'UD ', ud

            if numpy.mod(uidx, saveFreq) == 0:
                print 'Saving...',

                params = unzip(tparams)
                numpy.savez(saveto, history_errs=[], **params)
                pkl.dump(model_options, open('%s.pkl'%saveto, 'wb'))
                print 'Done'

        print 'Seen %d samples'%n_samples

    # Saving the model after all the epochs.
    print 'Saving...'

    params = unzip(tparams)
    numpy.savez(saveto, history_errs=[], **params)
    pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'))
    print 'Done'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=300)
    parser.add_argument('--decay-c', type=float, default=0.)
    parser.add_argument('--grad-clip', type=float, default=5.)
    parser.add_argument('--n-words', type=int)
    parser.add_argument('--maxlen-w', type=int, default=30)
    parser.add_argument('--dataset', type=str, default='amazon')
    parser.add_argument('--dictionary', type=str)
    parser.add_argument('--reload', action='store_true')
    parser.add_argument('--max-epochs', type=int, default=5)
    parser.add_argument('--encoder', type=str, default='gru')
    parser.add_argument('--saveto', type=str)

    args = parser.parse_args()

    trainer([], dim=args.dim, decay_c=args.decay_c, grad_clip=args.grad_clip, n_words=args.n_words,
            maxlen_w=args.maxlen_w, dataset=args.dataset, dictionary=args.dictionary, reload_=args.reload,
            max_epochs=args.max_epochs, encoder=args.encoder, saveto=args.saveto)
