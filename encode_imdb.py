import numpy
import os
import nltk
import pdb
import time
import skipthoughts
import cPickle
from training import tools
from scipy.io import loadmat

def preprocess(line, vocab):
    '''Perform the same kind of preprocessing as in the IMDB experiments.'''

    tokens = nltk.word_tokenize(line)
    tokenized_sentence = []
    for word in tokens:
        if word.lower() in vocab:
            tokenized_sentence.append(word.lower())

    return ' '.join(tokenized_sentence)

if __name__ == '__main__':

    # Set paths to the model.
    DATA_PATH = '/home/shunan/Code/Data/'
    VOCAB_FILE = os.path.join(DATA_PATH, 'word2vec/dict.txt')

    # The following directory should contain the data files.
    SAVE_PATH = os.path.join(DATA_PATH, 'imdb_sentiment/skip_thought_vecs/bi_skip_test.npy')

    # Loading the model and encoder.
    # model = skipthoughts.load_model()
    # encoder = skipthoughts.Encoder(model)
    model = tools.load_model(None)

    # loading the vocab
    vocab = set()
    vocab_f = open(VOCAB_FILE, 'r')
    for line in vocab_f:
        vocab.add(line.strip())

    # Load the movie review dataset.
    data = []
    # Do some preprocessing here as well.
    temp = loadmat(os.path.join(DATA_PATH, 'imdb_sentiment/imdb_sentiment.mat'))
    # encoding the test data
    if True:
        test_data = temp['test_data']
        test_labels = temp['test_labels']
        test_labels = test_labels.reshape([test_labels.shape[0]])

        for sen in test_data:
            sen = preprocess(sen[0][0].strip(), vocab)
            data.append(sen)
    if False:
        train_data = temp['train_data']
        train_labels = temp['train_labels']
        train_labels = train_labels.reshape([train_labels.shape[0]])

        # Only use the data that has labels.
        I = train_labels != 0
        train_labels_sup = train_labels[I]
        train_data_sup = train_data[I]

        for sen in train_data_sup:
            sen = preprocess(sen[0][0].strip(), vocab)
            data.append(sen)

    # Generate Skip-Thought Vectors for each sentence in the dataset.
    # encodings = encoder.encode(data)
    model_times = dict()
    for sen in data:
        sen_len = len(sen.split())
        encodings, time = tools.encode(model, [sen])
        if sen_len in model_times:
            model_times[sen_len].append(time)
        else:
            model_times[sen_len] = [time]

    # with open('./experiments/imdb/times_bi.pkl', 'w') as f:
    #     cPickle.dump(model_times, f)
