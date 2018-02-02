"""
Constructing and loading dictionaries
"""
import cPickle as pkl
import numpy
from collections import OrderedDict
from scipy.io import loadmat
import os
import nltk
from nltk.tokenize import TweetTokenizer

DATA_DIR = '/home/shunan/Code/Data'

def build_dictionary(text):
    """
    Build a dictionary
    text: list of sentences (pre-tokenized)
    """
    wordcount = OrderedDict()
    for cc in text:
        words = cc.split()
        for w in words:
            if w not in wordcount:
                wordcount[w] = 0
            wordcount[w] += 1
    words = wordcount.keys()
    freqs = wordcount.values()
    sorted_idx = numpy.argsort(freqs)[::-1]

    worddict = OrderedDict()
    # Words are indexed by decreasing order of frequency, for some reason.
    for idx, sidx in enumerate(sorted_idx):
        worddict[words[sidx]] = idx+2 # 0: <eos>, 1: <unk>

    return worddict, wordcount

def load_dictionary(loc='/ais/gobi3/u/rkiros/bookgen/book_dictionary_large.pkl'):
    """
    Load a dictionary
    """
    with open(loc, 'rb') as f:
        worddict = pkl.load(f)
    return worddict

def save_dictionary(worddict, wordcount, loc):
    """
    Save a dictionary to the specified location 
    """
    with open(loc, 'wb') as f:
        pkl.dump(worddict, f)
        pkl.dump(wordcount, f)


def build_dictionary_imdb():
    '''
    Build a dictionary, but using the indices to the word2vec data. We will also have to re-index the word2vec matrix.
    '''

    # Create the word to index mapping.
    word_to_index = dict()
    word2vec_dict_file = open(os.path.join(DATA_DIR, 'word2vec/dict.txt'), 'r')
    i = 0
    line = word2vec_dict_file.readline()
    while line != '':
        word_to_index[line.strip()] = i
        i += 1
        line = word2vec_dict_file.readline()

    word2vec_dict_file.close()

    imdb_data = loadmat(os.path.join(DATA_DIR, 'imdb_sentiment/imdb_sentiment.mat'))
    train_data = imdb_data['train_data']
    test_data = imdb_data['test_data']
    all_text = []

    for i in range(len(train_data)):
        line = train_data[i][0][0]
        tokens = nltk.word_tokenize(line)
        s = []
        for token in tokens:
            if token.lower() in word_to_index:
                s.append(token.lower())
        all_text.append(' '.join(s))

    for i in range(len(test_data)):
        line = train_data[i][0][0]
        tokens = nltk.word_tokenize(line)
        s = []
        for token in tokens:
            if token.lower() in word_to_index:
                s.append(token.lower())
        all_text.append(' '.join(s))

    worddict, wordcount = build_dictionary(all_text)

    # Re-indexing the word2vec vectors.
    word2vec = loadmat(os.path.join(DATA_DIR, 'word2vec/GoogleNews-vectors-negative300.mat'))
    word2vec = word2vec['vectors']
    new_matrix = numpy.random.uniform(-1, 1, (len(wordcount) + 2, 300))

    for word in worddict:
        old_ind = word_to_index[word]
        new_ind = worddict[word]
        vec = word2vec[old_ind, :]
        new_matrix[new_ind, :] = vec

    # Saving the new word2vec matrix.
    numpy.save('/home/shunan/Code/skip-thoughts/experiments/skip_thought_word2vec_embeds.npy', new_matrix)

    return worddict, wordcount

def build_dictionary_amazon():
    '''
    Similar to the above, except that this is for the Amazon dataset.
    '''

    word_to_index = dict()
    word2vec_dict_file = open(os.path.join(DATA_DIR, 'word2vec/dict.txt'), 'r')
    i = 0
    line = word2vec_dict_file.readline()
    while line != '':
        word_to_index[line.strip()] = i
        i += 1
        line = word2vec_dict_file.readline()

    word2vec_dict_file.close()

    with open(os.path.join(DATA_DIR, 'amazon_food/train_data.pkl')) as f:
        train_data = pkl.load(f)
        train_data = train_data[0]
    with open(os.path.join(DATA_DIR, 'amazon_food/test_data.pkl')) as f:
        test_data = pkl.load(f)
        test_data = test_data[0]
    tokenizer = TweetTokenizer()
    all_text = []

    for sen in train_data:
        sen = sen.strip().lower()
        tokens = nltk.word_tokenize(' '.join(tokenizer.tokenize(sen)))
        s = []
        for token in tokens:
            if token.lower() in word_to_index:
                s.append(token.lower())
        all_text.append(' '.join(s))

    for sen in test_data:
        sen = sen.strip().lower()
        tokens = nltk.word_tokenize(' '.join(tokenizer.tokenize(sen)))
        s = []
        for token in tokens:
            if token.lower() in word_to_index:
                s.append(token.lower())
        all_text.append(' '.join(s))

    worddict, wordcount = build_dictionary(all_text)
    print('Number of words: {}'.format(wordcount))

    # Re-indexing the word2vec vectors.
    word2vec = loadmat(os.path.join(DATA_DIR, 'word2vec/GoogleNews-vectors-negative300.mat'))
    word2vec = word2vec['vectors']
    new_matrix = numpy.random.uniform(-1, 1, (len(wordcount) + 2, 300))

    for word in worddict:
        old_ind = word_to_index[word]
        new_ind = worddict[word]
        vec = word2vec[old_ind, :]
        new_matrix[new_ind, :] = vec

    # Saving the new word2vec matrix.
    numpy.save('/home/shunan/Code/skip-thoughts/experiments/amazon/word2vec_embeds.npy', new_matrix)

    return worddict, wordcount