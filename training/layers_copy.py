"""
Layers for skip-thoughts
To add a new layer:
1) Add layer names to the 'layers' dictionary below
2) Implement param_init and feedforward functions
3) In the trainer function, replace 'encoder' or 'decoder' with your layer name
"""
import theano
import theano.tensor as tensor

import numpy

from utils import _p, ortho_weight, norm_weight, tanh, linear # note the activation functions

# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'gru': ('param_init_gru', 'gru_layer'),
          'zoneout_gru': ('param_init_gru', 'zoneout_gru_layer'),
          'lngru': ('param_init_lngru', 'lngru_layer'),
          'lstm_peep': ('param_init_lstm_peep', 'lstm_peep_layer'),
          'lstm': ('param_init_lstm', 'lstm_layer'),
          'lnlstm': ('param_init_lnlstm', 'lnlstm_layer'),
          'bidirectional': ('param_init_bidirectional', 'bidirectional_layer'),
          'char_level': ('param_init_char_level', 'char_level_layer'),
          'char_level_shared': ('param_init_char_level_shared', 'char_level_layer_shared'),
          'bag_of_words': ('param_init_bag_of_words', 'bag_of_words_layer'),
          'bigram': ('param_init_bigram', 'bigram_layer')
          }


def get_layer(name):
    """
    Return param init and feedforward functions for the given layer name
    """
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))


# layer normalization
def ln(x, b, s):
    _eps = 1e-5
    output = (x - x.mean(1)[:,None]) / tensor.sqrt((x.var(1)[:,None] + _eps))
    output = s[None, :] * output + b[None,:]
    return output


# Feedforward layer
def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None, ortho=True):
    """
    Affine transformation + point-wise non-linearity
    """
    if nin == None:
        nin = options['dim_proj']
    if nout == None:
        nout = options['dim_proj']
    params[_p(prefix,'W')] = norm_weight(nin, nout, ortho=ortho)
    params[_p(prefix,'b')] = numpy.zeros((nout,)).astype('float32')

    return params


def fflayer(tparams, state_below, options, prefix='rconv', activ='lambda x: tensor.tanh(x)', **kwargs):
    """
    Feedforward pass
    """
    return eval(activ)(tensor.dot(state_below, tparams[_p(prefix,'W')]) + tparams[_p(prefix,'b')])


# GRU layer
def param_init_gru(options, params, prefix='gru', nin=None, dim=None):
    """
    Gated Recurrent Unit (GRU)
    The following equations defne GRU
    u = sig(x_t Wu + h_t-1 Uu + bu)
    r = sig(x_t Wr + h_t-1 Ur + br)
    h = tanh(x_t Wx + (s_t-1 . r) Ux + bx)
    s_t = (1 - u) . h + u . s_t-1
    Below some of the parameters are initlaized together and later sliced
    W = [Wu Wr], i.e. the (horizontal) concatination of Wu and Wr
    b = [bu br]
    U = [Uu Ur]
    """
    if nin == None:
        nin = options['dim_word']
    if dim == None:
        dim = options['dim_proj']
    W = numpy.concatenate([norm_weight(nin,dim),
                           norm_weight(nin,dim)], axis=1)
    params[_p(prefix,'W')] = W
    params[_p(prefix,'b')] = numpy.zeros((2 * dim,)).astype('float32')
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix,'U')] = U

    Wx = norm_weight(nin, dim)
    params[_p(prefix,'Wx')] = Wx
    Ux = ortho_weight(dim)
    params[_p(prefix,'Ux')] = Ux
    params[_p(prefix,'bx')] = numpy.zeros((dim,)).astype('float32')

    return params


def gru_layer(tparams, state_below, init_state, options, prefix='gru', mask=None, go_backwards=False, **kwargs):
    """
    Feedforward pass through GRU
    state_below = x_t [Wu Wr] + [bu br]
    state_belowx = x_t Ux + bx
    """
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    dim = tparams[_p(prefix,'Ux')].shape[1]

    if init_state == None:
        init_state = tensor.alloc(0., n_samples, dim)

    if mask == None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # how do these lines make sense when W is not the same size as Wx?
    # because its a horizontal concatinate in the init
    # These can be precomputed because we know the input, i.e. state_below
    # and then given as a sequence to the scan function
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + tparams[_p(prefix, 'bx')]
    # U = tparams[_p(prefix, 'U')]
    # Ux = tparams[_p(prefix, 'Ux')]

    def _step_slice(m_, x_, xx_, h_, U, Ux): # i.e. mask, state_below_, state_belowx, h, U, Ux
        preact = tensor.dot(h_, U) #compute the hidden to hidden weight vector
        preact += x_

        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        preactx = preactx + xx_ # xx_ = x_t Wx

        h = tensor.tanh(preactx) # h = tanh(x_t Wx + (s_t-1 . r) Ux + bx)

        h = u * h_ + (1. - u) * h
        h = m_[:,None] * h + (1. - m_)[:,None] * h_

        return h

    seqs = [mask, state_below_, state_belowx]
    _step = _step_slice

    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info = [init_state],
                                non_sequences = [tparams[_p(prefix, 'U')],
                                                 tparams[_p(prefix, 'Ux')]],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                go_backwards=go_backwards,
                                profile=False,
                                strict=True)
    return [rval]


def zoneout_gru_layer(tparams, state_below, init_state, options, prefix='gru', mask=None, go_backwards=False, **kwargs):
    """
    Zoneout GRU based on code from https://github.com/teganmaharaj/zoneout/blob/master/zoneout_theano.py
    paper: https://arxiv.org/pdf/1606.01305v2.pdf
    Feedforward pass through GRU
    state_below = x_t [Wu Wr] + [bu br]
    state_belowx = x_t Ux + bx
    """

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    dim = tparams[_p(prefix,'Ux')].shape[1]

    if init_state == None:
        init_state = tensor.alloc(0., n_samples, dim)

    try:
        z_prob_states = options['z_prob_states']
    except:
        z_prob_states = 0.05

    try:
        is_test_time = options['is_test_time']
    except:
        is_test_time = False

    if is_test_time:
        zoneouts_states = tensor.cast(numpy.ones(state_below.shape) * (1-z_prob_states), theano.config.floatX)
    else:
        srng = theano.tensor.shared_randomstreams.RandomStreams(numpy.random.RandomState(0).randint(999999))
        zoneouts_states = srng.binomial(n=1, p=(1-z_prob_states), size=(state_below.shape[0], state_below.shape[1], dim), dtype=theano.config.floatX)

    if mask == None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # how do these lines make sense when W is not the same size as Wx?
    # because its a horizontal concatinate in the init
    # These can be precomputed because we know the input, i.e. state_below
    # and then given as a sequence to the scan function
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + tparams[_p(prefix, 'bx')]
    # U = tparams[_p(prefix, 'U')]
    # Ux = tparams[_p(prefix, 'Ux')]

    def _step_slice(zoneouts_states, m_, x_, xx_, h_, U, Ux): # i.e. mask, state_below_, state_belowx, h, U, Ux
        preact = tensor.dot(h_, U) #compute the hidden to hidden weight vector
        preact += x_

        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        preactx = preactx + xx_ # xx_ = x_t Wx

        h = tensor.tanh(preactx) # h = tanh(x_t Wx + (s_t-1 . r) Ux + bx)

        h = u * h_ + (1. - u) * h * (1 - zoneouts_states)  # igates_zoneouts

        h = h_ * zoneouts_states + (1 - zoneouts_states) * h  # zone out based on previous state h_

        h = m_[:,None] * h + (1. - m_)[:,None] * h_

        return h

    seqs = [zoneouts_states, mask, state_below_, state_belowx]
    _step = _step_slice

    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info = [init_state],
                                non_sequences = [tparams[_p(prefix, 'U')],
                                                 tparams[_p(prefix, 'Ux')]],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                go_backwards=go_backwards,
                                profile=False,
                                strict=True)
    return [rval]


# LSTM layer with Peepholes
def param_init_lstm_peep(options, params, prefix='lstm', nin=None, dim=None):
    """
    Code based on http://deeplearning.net/tutorial/code/lstm.py and Jamie's GRU code
    Long Short Term Memory Unit (LSTM)
    LSTM is defined by the follow equations,
    W = [Wi Wf Wc Wo] # input weights
    b = [bi bf bc bo] # biases
    U = [Ui Uf Uc Uo] # recurrent weights
    Pi Pf Po c_t-1    # peep hole params and the previous cell, c_t-1
    i_t = sig(Wi x_t + Ui h_t-1 + Pi c_t-1 + bi)
    f_t = sig(Wf x_t + Uf h_t-1 + Pf c_t-1 + bf)
    c_t = f_t c_t-1 + i_t tanh(Wc x_t + Uc h_t-1 + bc)
    o_t = sig(Wo x_t + Uo h_t-1 + Po c_t-1 + bo)
    h_t = o_t tanh(c_t)
    """
    if nin == None:
        nin = options['dim_word']
    if dim == None:
        dim = options['dim_proj']

    # input weight matrix is 4 times for the input gate, forget gate, output gate, and cell input
    W = numpy.concatenate([norm_weight(nin,dim), norm_weight(nin,dim),
                           norm_weight(nin,dim), norm_weight(nin,dim)], axis=1)
    params[_p(prefix,'W')] = W
    params[_p(prefix,'b')] = numpy.zeros((4 * dim,)).astype('float32')

    # The recurrent weight matrix
    U = numpy.concatenate([ortho_weight(dim), ortho_weight(dim),  # remember this is ortho_weight(dim, dim)
                           ortho_weight(dim), ortho_weight(dim)], axis=1)
    params[_p(prefix,'U')] = U

    # Peep holes weight vectors, all initialized to zero
    # Peep hole weights are diagonal as in Grave's paper
    params[_p(prefix,'Pi')] = numpy.zeros((dim,)).astype('float32')
    params[_p(prefix,'Pf')] = numpy.zeros((dim,)).astype('float32')
    params[_p(prefix,'Po')] = numpy.zeros((dim,)).astype('float32')

    # inital h_0, and cell get made in lstm_layer or passed in

    # initialize forget gates to one?
    return params


def lstm_peep_layer(tparams, state_below, init_state, options, prefix='lstm', mask=None, go_backwards=False, c = None, **kwargs):
    """
    Feedforward pass through LSTM
    """
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # dim = n_hidden, i.e. size of cell = n_i = n_c = n_o = n_f
    dim = tparams[_p(prefix,'U')].shape[0]  # or .shape[1] // 4

    if not init_state:
        init_state = tensor.alloc(0.0, n_samples, dim)

    if not c:
        c = tensor.alloc(0.0, n_samples, dim)  # (value, *shape)

    if not mask:
        mask = tensor.alloc(1.0, state_below.shape[0], 1)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # state_below = W x_t + b
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]

    def _step_slice(m_, x_, h_, c_, U, Pi, Pf, Po):  # i.e. mask, state_below_, h, previous cell, U, Pi, Pf, Po
        preact = tensor.dot(h_, U)  # U
        preact += x_  # W x_t + U h_t-1 + b

        # i_t = sig(Wi x_t + Ui h_t-1 + Pi c_t-1 + bi)
        # f_t = sig(Wf x_t + Uf h_t-1 + Pf c_t-1 + bf)
        # c_t = f_t c_t-1 + i_t tanh(Wc x_t + Uc h_t-1 + bc)
        # o_t = sig(Wo x_t + Uo h_t-1 + Po c_t-1 + bo)
        # h_t = o_t tanh(c_t)

        i = tensor.nnet.sigmoid(_slice(preact, 0, dim) + c_ * Pi)
        f = tensor.nnet.sigmoid(_slice(preact, 1, dim) + c_ * Pf)
        c = f * c_ + i * tensor.tanh(_slice(preact, 2, dim))
        o = tensor.nnet.sigmoid(_slice(preact, 3, dim) + c_ * Po)

        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    seqs = [mask, state_below_]
    _step = _step_slice

    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=[init_state, c],
                                non_sequences=[tparams[_p(prefix, 'U')], tparams[_p(prefix, 'Pi')], tparams[_p(prefix, 'Pf')], tparams[_p(prefix, 'Po')]],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=False,
                                go_backwards=go_backwards,
                                strict=True)
    return [rval[0]] # only return hidden layer, not cell


# LN-GRU layer
def param_init_lngru(options, params, prefix='lngru', nin=None, dim=None):
    """
    Gated Recurrent Unit (GRU) with LN
    """
    if nin == None:
        nin = options['dim_proj']
    if dim == None:
        dim = options['dim_proj']
    W = numpy.concatenate([norm_weight(nin,dim),
                           norm_weight(nin,dim)], axis=1)
    params[_p(prefix,'W')] = W
    params[_p(prefix,'b')] = numpy.zeros((2 * dim,)).astype('float32')
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix,'U')] = U

    Wx = norm_weight(nin, dim)
    params[_p(prefix,'Wx')] = Wx
    Ux = ortho_weight(dim)
    params[_p(prefix,'Ux')] = Ux
    params[_p(prefix,'bx')] = numpy.zeros((dim,)).astype('float32')

    # LN parameters
    scale_add = 0.0
    scale_mul = 1.0
    params[_p(prefix,'b1')] = scale_add * numpy.ones((2*dim)).astype('float32')
    params[_p(prefix,'b2')] = scale_add * numpy.ones((1*dim)).astype('float32')
    params[_p(prefix,'b3')] = scale_add * numpy.ones((2*dim)).astype('float32')
    params[_p(prefix,'b4')] = scale_add * numpy.ones((1*dim)).astype('float32')
    params[_p(prefix,'s1')] = scale_mul * numpy.ones((2*dim)).astype('float32')
    params[_p(prefix,'s2')] = scale_mul * numpy.ones((1*dim)).astype('float32')
    params[_p(prefix,'s3')] = scale_mul * numpy.ones((2*dim)).astype('float32')
    params[_p(prefix,'s4')] = scale_mul * numpy.ones((1*dim)).astype('float32')

    return params

def lngru_layer(tparams, state_below, init_state, options, prefix='lngru', mask=None, go_backwards=False, **kwargs):
    """
    Feedforward pass through GRU with LN
    """
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    dim = tparams[_p(prefix,'Ux')].shape[1]

    if init_state == None:
        init_state = tensor.alloc(0., n_samples, dim)

    if mask == None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + tparams[_p(prefix, 'bx')]
    U = tparams[_p(prefix, 'U')]
    Ux = tparams[_p(prefix, 'Ux')]

    def _step_slice(m_, x_, xx_, h_, U, Ux, b1, b2, b3, b4, s1, s2, s3, s4):

        x_ = ln(x_, b1, s1)
        xx_ = ln(xx_, b2, s2)

        preact = tensor.dot(h_, U)
        preact = ln(preact, b3, s3)
        preact += x_

        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        preactx = tensor.dot(h_, Ux)
        preactx = ln(preactx, b4, s4)
        preactx = preactx * r
        preactx = preactx + xx_

        h = tensor.tanh(preactx)

        h = u * h_ + (1. - u) * h
        h = m_[:,None] * h + (1. - m_)[:,None] * h_

        return h

    seqs = [mask, state_below_, state_belowx]
    _step = _step_slice

    non_seqs = [tparams[_p(prefix, 'U')], tparams[_p(prefix, 'Ux')]]
    non_seqs += [tparams[_p(prefix, 'b1')], tparams[_p(prefix, 'b2')], tparams[_p(prefix, 'b3')], tparams[_p(prefix, 'b4')]]
    non_seqs += [tparams[_p(prefix, 's1')], tparams[_p(prefix, 's2')], tparams[_p(prefix, 's3')], tparams[_p(prefix, 's4')]]

    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info = [init_state],
                                non_sequences = non_seqs,
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=False,
                                go_backwards=go_backwards,
                                strict=True)
    rval = [rval]
    return rval

# LSTM layer init
def param_init_lstm(options, params, prefix='lstm', nin=None, dim=None):
    if nin is None:
        nin = options['dim_proj']

    if dim is None:
        dim = options['dim_proj']

    W = numpy.concatenate([norm_weight(nin,dim),norm_weight(nin,dim),
                           norm_weight(nin,dim),norm_weight(nin,dim)], axis=1)
    params[_p(prefix,'W')] = W
    U = numpy.concatenate([ortho_weight(dim), ortho_weight(dim),
                           ortho_weight(dim), ortho_weight(dim)],axis=1)
    params[_p(prefix,'U')] = U
    params[_p(prefix,'b')] = numpy.zeros((4 * dim,)).astype('float32')

    return params

# LSTM layer
def lstm_layer(tparams, state_below, init_state, options, prefix='lstm', mask=None, go_backwards=False, c = None, **kwargs):

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    param = lambda name: tparams[_p(prefix, name)]
    dim = param('U').shape[0]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)


    U = param('U')
    b = param('b')
    W = param('W')
    non_seqs = [U, b, W]

    # initial/previous memory
    if c is None:
        c = tensor.alloc(0., n_samples, dim)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    def _step(mask, sbelow, sbefore, cell_before, *args):
        preact =  tensor.dot(sbefore, param('U'))
        preact += sbelow
        preact += param('b')

        i = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        f = tensor.nnet.sigmoid(_slice(preact, 1, dim))
        o = tensor.nnet.sigmoid(_slice(preact, 2, dim))
        c = tensor.tanh(_slice(preact, 3, dim))

        c = f * cell_before + i * c
        c = mask * c + (1. - mask) * cell_before
        h = o * tensor.tanh(c)
        h = mask * h + (1. - mask) * sbefore

        return h, c

    lstm_state_below =  tensor.dot(state_below, param('W')) + param('b')
    if state_below.ndim == 3:
        lstm_state_below = lstm_state_below.reshape((state_below.shape[0], state_below.shape[1], -1))

    if mask.ndim == 3 and mask.ndim == state_below.ndim:
        mask = mask.reshape((mask.shape[0], mask.shape[1]*mask.shape[2])).dimshuffle(0, 1, 'x')
    elif mask.ndim == 2:
        mask = mask.dimshuffle(0, 1, 'x')

    rval, updates = theano.scan(_step,
                                sequences=[mask, lstm_state_below],
                                outputs_info = [init_state, c],
                                name=_p(prefix, '_layers'),
                                non_sequences=non_seqs,
                                strict=True,
                                go_backwards=go_backwards,
                                n_steps=nsteps)
    return rval

# LN-LSTM init
def param_init_lnlstm(options, params, prefix='lnlstm', nin=None, dim=None):
    if nin is None:
        nin = options['dim_proj']

    if dim is None:
        dim = options['dim_proj']

    W = numpy.concatenate([norm_weight(nin,dim), norm_weight(nin,dim),
                           norm_weight(nin,dim), norm_weight(nin,dim)], axis=1)
    params[_p(prefix,'W')] = W
    U = numpy.concatenate([ortho_weight(dim), ortho_weight(dim),
                           ortho_weight(dim), ortho_weight(dim)],axis=1)
    params[_p(prefix,'U')] = U
    params[_p(prefix,'b')] = numpy.zeros((4 * dim,)).astype('float32')

    # lateral parameters
    scale_add = 0.0
    scale_mul = 1.0
    params[_p(prefix,'b1')] = scale_add * numpy.ones((4*dim)).astype('float32')
    params[_p(prefix,'b2')] = scale_add * numpy.ones((4*dim)).astype('float32')
    params[_p(prefix,'b3')] = scale_add * numpy.ones((1*dim)).astype('float32')
    params[_p(prefix,'s1')] = scale_mul * numpy.ones((4*dim)).astype('float32')
    params[_p(prefix,'s2')] = scale_mul * numpy.ones((4*dim)).astype('float32')
    params[_p(prefix,'s3')] = scale_mul * numpy.ones((1*dim)).astype('float32')

    return params

# LN-LSTM layer
def lnlstm_layer(tparams, state_below, init_state, options, prefix='lnlstm',
                 mask=None, go_backwards=False, c = None, **kwargs):

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    param = lambda name: tparams[_p(prefix, name)]
    dim = param('U').shape[0]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    U = param('U')
    b = param('b')
    W = param('W')
    non_seqs = [U, b, W]
    non_seqs.extend(list(map(param, "b1 b2 b3 s1 s2 s3".split())))

    # initial/previous memory
    if c is None:
        c = tensor.alloc(0., n_samples, dim)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    def _step(mask, sbelow, sbefore, cell_before, *args):
        sbelow_ = ln(sbelow, param('b1'), param('s1'))
        sbefore_ = ln( tensor.dot(sbefore, param('U')), param('b2'), param('s2'))

        preact = sbefore_ + sbelow_ + param('b')

        i = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        f = tensor.nnet.sigmoid(_slice(preact, 1, dim))
        o = tensor.nnet.sigmoid(_slice(preact, 2, dim))
        c = tensor.tanh(_slice(preact, 3, dim))

        c = f * cell_before + i * c
        c = mask * c + (1. - mask) * cell_before

        c_ = ln(c, param('b3'), param('s3'))
        h = o * tensor.tanh(c_)
        h = mask * h + (1. - mask) * sbefore

        return h, c

    lstm_state_below =  tensor.dot(state_below, param('W')) + param('b')
    if state_below.ndim == 3:
        lstm_state_below = lstm_state_below.reshape((state_below.shape[0],
                                                     state_below.shape[1],
                                                     -1))

    if mask.ndim == 3 and mask.ndim == state_below.ndim:
        mask = mask.reshape((mask.shape[0], mask.shape[1]*mask.shape[2])).dimshuffle(0, 1, 'x')
    elif mask.ndim == 2:
        mask = mask.dimshuffle(0, 1, 'x')

    rval, updates = theano.scan(_step,
                                sequences=[mask, lstm_state_below],
                                outputs_info = [init_state, c],
                                name=_p(prefix, '_layers'),
                                non_sequences=non_seqs,
                                strict=True,
                                go_backwards=go_backwards,
                                n_steps=nsteps)
    return rval


# Bidirectional layer
def param_init_bidirectional(options, params, layer_type='lstm', prefix='bidi_lstm', nin=None, dim=None):
    params = get_layer(layer_type)[0](options, params, prefix=prefix+'_forward', nin=nin, dim=dim)
    params = get_layer(layer_type)[0](options, params, prefix=prefix+'_backward', nin=nin, dim=dim)
    return params


def bidirectional_layer(tparams, state_below, init_state, options, layer_type='lstm', prefix='bidi_lstm',
                        mask=None, c=None, concat_projections=True, **kwargs):
    proj = get_layer(layer_type)[1](tparams, state_below, init_state, options, prefix=prefix+'_forward',
                                    mask=mask, go_backwards=False, c=c **kwargs)

    proj_r = get_layer(layer_type)[1](tparams, state_below, init_state, options, prefix=prefix+'_backward',
                                      mask=mask, go_backwards=True, c=c, **kwargs)

    if concat_projections:
        return tensor.concatenate([proj[0], proj_r[0][::-1]], axis=proj[0].ndim-1)
    else:
        return (proj + proj_r) / 2 # mean, not sum


#Char level
def param_init_char_level(options, params, layer_type='lstm', prefix='char_level_lstm', char_dim=None, word_dim=None, sent_dim=None):
    params = get_layer(layer_type)[0](options, params, prefix=prefix+'_char_to_word', nin=char_dim, dim=word_dim)
    params = get_layer(layer_type)[0](options, params, prefix=prefix+'_word_to_sent', nin=word_dim, dim=sent_dim)
    return params


def char_level_layer(tparams, state_below, init_state, options, layer_type='lstm', prefix='char_level_lstm', mask=None, c=None, **kwargs):
    original_shape = state_below.shape
    original_mask_shape = mask.shape
    state_below = state_below.reshape([original_shape[0], original_shape[1]*original_shape[2], original_shape[3]])
    mask = mask.reshape([original_shape[0], original_shape[1]*original_shape[2]])

    word_proj = get_layer(layer_type)[1](tparams, state_below, init_state, options, prefix=prefix+'_char_to_word',
                                    mask=mask, c=c, **kwargs)

    word_dim = word_proj[0].shape[1]
    word_proj = word_proj[0][-1,:].reshape([original_shape[1], original_shape[2], word_dim])
    mask = mask.reshape(original_mask_shape).max() # max over first dimension
    sent_proj = get_layer(layer_type)[1](tparams, word_proj, init_state, options, prefix=prefix+'_word_to_sent',
                                      mask=mask, c=c, **kwargs)

    return sent_proj


# Bag of words compositional function
def param_init_bag_of_words(options, params, prefix='bow', nin=None, dim=None):
    """
    Average of word vectors
    """
    return params


def bag_of_words_layer(tparams, state_below, init_state, options, prefix='bow', mask=None, go_backwards=False, **kwargs):
    """
    Average of word vectors f
    """
    if mask == None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    emb = (state_below * mask[:, :, None]).sum(axis=0) / mask.sum(axis=0)[:, None]
    rval = emb
    return [[rval]]  # need to treat it as a list of timesteps for encoder API


# Bigram compositional function
def param_init_bigram(options, params, prefix='bigram', nin=None, dim=None):
    """
    1/n Sum_i tanh(a_i + a_i+1)
    """
    return params


def bigram_layer(tparams, state_below, init_state, options, prefix='bigram', mask=None, go_backwards=False, **kwargs):
    """
    1/n Sum_i tanh(a_i + a_i+1)
    """
    nsteps = state_below.shape[0]
    n_samples = state_below.shape[1]
    dims = state_below.shape[2]


    if mask == None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    init_state = tensor.alloc(0., n_samples, dims)

    def _step_slice(m_, x_, h_):  # i.e. mask, a_next, a_cur

        h = tensor.tanh(h_ + x_)
        h = m_[:, None] * h + (1. - m_[:, None]) * h_  # copy over hidden state
        return h

    seqs = [mask, state_below]
    _step = _step_slice

    # Calculate bigrams
    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=[init_state],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                go_backwards=go_backwards,
                                profile=False,
                                strict=True)
    # Do sum
    emb = rval
    emb = (emb * mask[:, :, None]).sum(axis=0) / mask.sum(axis=0)[:, None]
    rval = emb

    return [[rval]]