import numpy as np
import os
import skipthoughts
import sqlite3
from nltk.tokenize import TweetTokenizer
import nltk
import cPickle
from training import tools
import time
import pdb

DATA_PATH = '/home/shunan/Code/Data/'

def get_data(word_set):
    '''
    Get all the data and return it as a list.
    '''

    conn = sqlite3.connect(os.path.join(DATA_PATH, 'amazon_food/database.sqlite'))
    c = conn.cursor()
    all_data = []
    tokenizer = TweetTokenizer()
    for row in c.execute('SELECT Score, Text FROM Reviews'):
        sen = row[1].strip().lower()
        tmp = nltk.word_tokenize(' '.join(tokenizer.tokenize(sen)))
        new_sen = []
        for tok in tmp:
            if tok in word_set:
                new_sen.append(tok)
        all_data.append(' '.join(new_sen))

    return all_data

def preprocess(data, word_set):
    '''
    Preprocess list of documents.
    '''

    preprocessed = []
    tokenizer = TweetTokenizer()
    for elem in data:
        sen = elem.strip().lower()
        tmp = nltk.word_tokenize(' '.join(tokenizer.tokenize(sen)))
        new_sen = [tok for tok in tmp if tok in word_set]
        preprocessed.append(' '.join(new_sen))
    return preprocessed

def get_pretrained_encodings(pretrained=False):
    '''
    Get encodings using the pre-trained models.
    '''

    word_set = set()
    dict_f = open(os.path.join(DATA_PATH, 'word2vec/dict.txt'), 'r')
    for line in dict_f:
        word_set.add(line.strip())
    dict_f.close()

    # Getting the data.
    with open(os.path.join(DATA_PATH, 'amazon_food/train_data.pkl'), 'r') as f:
        train_data = cPickle.load(f)
        train_preprocessed = preprocess(train_data[0], word_set)
    with open(os.path.join(DATA_PATH, 'amazon_food/test_data.pkl'), 'r') as f:
        test_data = cPickle.load(f)
        test_preprocessed = preprocess(test_data[0], word_set)

    if pretrained:
        model = skipthoughts.load_model()
        encoder = skipthoughts.Encoder(model)
        test_save_path = os.path.join(DATA_PATH, 'amazon_food/skip_thought_vecs/skip_thought_vecs_test_pretrained.npy')
        train_save_path = os.path.join(DATA_PATH,
                                       'amazon_food/skip_thought_vecs/skip_thought_vecs_train_pretrained.npy')
        print('Encoding training vectors')
        train_vectors = encoder.encode(train_preprocessed)
        print('Encoding test vectors')
        test_vectors = encoder.encode(test_preprocessed)
    else:
        model = tools.load_model(None)
        test_save_path = os.path.join(DATA_PATH, 'amazon_food/skip_thought_vecs/skip_thought_vecs_test_bi.npy')
        train_save_path = os.path.join(DATA_PATH, 'amazon_food/skip_thought_vecs/skip_thought_vecs_train_bi.npy')
        print('Encoding training vectors')
        train_vectors = tools.encode(model, train_preprocessed)
        print('Encoding test vectors')
        test_vectors = tools.encode(model, test_preprocessed)

    np.save(train_save_path, train_vectors)
    np.save(test_save_path, test_vectors)

if __name__ == '__main__':

    get_pretrained_encodings(True)
