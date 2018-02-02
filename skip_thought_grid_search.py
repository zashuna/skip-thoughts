import sys
import itertools
sys.path.append('/home/shunan/Code/CNN_Doc2Vec/imdb')
sys.path.append('/home/shunan/Code/CNN_Doc2Vec/Amazon_Doc2Vec')
import imdb_experiments
import amazon_experiments
import os
import cPickle
import subprocess
import numpy as np
from training import train
from training import tools
from scipy.io import loadmat
from encode_amazon import preprocess as amazon_preprocess
from encode_imdb import preprocess as imdb_preprocess
import pdb

DATA_PATH = '/home/shunan/Code/Data/'
MAX_EPOCHS = 5

# Script to do a grid search across the different parameters.
def generate_param_combinations(hyper_params):
    '''Generate and return all combinations of the hyper parameters for the grid search.'''

    all_params = sorted(hyper_params)
    all_combs = itertools.product(*(hyper_params[name] for name in all_params))
    all_combs = list(all_combs)

    combinations_list = []
    for comb in all_combs:
        d = dict(zip(all_params, comb))
        combinations_list.append(d)

    return combinations_list


def get_data(dataset):
    '''Get the data that is to be encoded.'''

    word_set = set()
    dict_f = open(os.path.join(DATA_PATH, 'word2vec/dict.txt'), 'r')
    for line in dict_f:
        word_set.add(line.strip())
    dict_f.close()

    if dataset == 'amazon':
        # Getting the data.
        with open(os.path.join(DATA_PATH, 'amazon_food/train_data.pkl'), 'r') as f:
            train_data_all = cPickle.load(f)
            train_labels = np.array(train_data_all[1]) - 1
            train_data = train_data_all[0]
        with open(os.path.join(DATA_PATH, 'amazon_food/test_data.pkl'), 'r') as f:
            test_data_all = cPickle.load(f)
            test_labels = np.array(test_data_all[1]) - 1
            test_data = test_data_all[0]

        # binarizing the data
        I = train_labels != 3
        train_labels_bin = train_labels[I] >= 4
        train_vecs_bin = []
        for i in range(len(I)):
            if I[i]:
                train_vecs_bin.append(train_data[i])
        I = test_labels != 3
        test_labels_bin = test_labels[I] >= 4
        test_vecs_bin = []
        for i in range(len(I)):
            if I[i]:
                test_vecs_bin.append(test_data[i])
        train_preprocessed = amazon_preprocess(train_vecs_bin, word_set)
        test_preprocessed = amazon_preprocess(test_vecs_bin, word_set)

        return train_preprocessed, train_labels_bin, test_preprocessed, test_labels_bin
    elif dataset == 'imdb':
        train_preprocessed, test_preprocessed = [], []
        temp = loadmat(os.path.join(DATA_PATH, 'imdb_sentiment/imdb_sentiment.mat'))

        # Grabbing the test data first
        test_data = temp['test_data']
        for sen in test_data:
            sen = imdb_preprocess(sen[0][0].strip(), word_set)
            test_preprocessed.append(sen)

        # Grabbing the training data
        train_data = temp['train_data']
        train_labels = temp['train_labels']
        train_labels = train_labels.reshape([train_labels.shape[0]])
        # Only use the data that has labels.
        I = train_labels != 0
        train_labels_sup = train_labels[I]
        train_data_sup = train_data[I]

        test_labels = temp['test_labels']
        test_labels = test_labels.reshape([test_labels.shape[0]])

        for sen in train_data_sup:
            sen = imdb_preprocess(sen[0][0].strip(), word_set)
            train_preprocessed.append(sen)

        test_labels_sup = test_labels >= 7
        train_labels_sup = train_labels_sup >= 7

        return train_preprocessed, train_labels_sup, test_preprocessed, test_labels_sup
    else:
        return None


def call_training(param, n_words, dataset, dict_loc, reload_, encoder, save_loc):
    '''
    Train the skip-thought model as a subprocess.
    '''

    subprocess_call = ['python', './training/train.py']
    for option in param:
        subprocess_call.append('--' + option)
        subprocess_call.append(str(param[option]))

    additional_params = ['--n-words', str(n_words), '--dataset', dataset, '--dictionary', dict_loc, '--encoder',
                         encoder, '--saveto', save_loc, '--max-epochs', '1']
    if reload_:
        additional_params.append('--reload')

    subprocess_call.extend(additional_params)
    subprocess.call(subprocess_call)


def run_grid_search(hyper_params, dataset):
    '''
    Run the grid search experiments, given the hyper-parameters
    '''

    all_params = generate_param_combinations(hyper_params)
    if dataset == 'amazon':
        n_words = 38830
        dict_location = '/home/shunan/Code/skip-thoughts/experiments/amazon/word_dicts.pkl'
    elif dataset == 'imdb':
        n_words = 64526
        dict_location = '/home/shunan/Code/skip-thoughts/experiments/imdb/word_dicts.pkl'

    exp_info = {
        'dataset': dataset,
        'param_num': 0,
        'epoch_num': 0,
        'max_acc_uni': 0,
        'max_acc_uni_params': None,
        'max_acc_bi': 0,
        'max_acc_bi_params': None,
        'max_acc_combine': 0,
        'max_acc_combine_params': None
    }
    uni_save_loc = '/home/shunan/Code/skip-thoughts/experiments/{}/model_uni.npz'.format(dataset)
    bi_save_loc = '/home/shunan/Code/skip-thoughts/experiments/{}/model_bi.npz'.format(dataset)
    dict_path = '/home/shunan/Code/skip-thoughts/experiments/{}/word_dicts.pkl'.format(dataset)

    # Getting the data to encode, not for training.
    train_data, train_labels, test_data, test_labels = get_data(dataset)

    for p in range(len(all_params)):
        # loading from previous grid search.
        if p < 5:
            continue
        elif p == 5:
            load = False
            e = 0
        else:
            load = False
            e = 0

        param = all_params[p]
        print('Using hyper-parameter setting {} of {}'.format(p + 1, len(all_params)))
        exp_info['param_num'] = p
        while e < MAX_EPOCHS:
            exp_info['epoch_num'] = e

            call_training(param, n_words, dataset, dict_location, load, 'gru', uni_save_loc)
            print('Training bidirectional model.')
            call_training(param, n_words, dataset, dict_location, load, 'bidirectional', bi_save_loc)
            if e == 0:
                load = True

            # Running the classification experiment.
            model_uni = tools.load_model(path_to_model=uni_save_loc, path_to_dictionary=dict_path)
            model_bi = tools.load_model(path_to_model=bi_save_loc, path_to_dictionary=dict_path)

            print('Encoding uni-directional vectors')
            uni_train_vectors = tools.encode(model_uni, train_data)
            uni_test_vectors = tools.encode(model_uni, test_data)
            print('Encoding bi-directional vectors')
            bi_train_vectors = tools.encode(model_bi, train_data)
            bi_test_vectors = tools.encode(model_bi, test_data)
            combine_train_vectors = np.hstack((uni_train_vectors, bi_train_vectors))
            combine_test_vectors = np.hstack((uni_test_vectors, bi_test_vectors))

            # Training the classifier now.
            if dataset == 'amazon':
                acc = amazon_experiments.pre_trained_experiments(uni_train_vectors, train_labels, uni_test_vectors,
                                                                 test_labels, 2)
                if acc > exp_info['max_acc_uni']:
                    exp_info['max_acc_uni'] = acc
                    exp_info['max_acc_uni_params'] = param
                acc = amazon_experiments.pre_trained_experiments(bi_train_vectors, train_labels, bi_test_vectors,
                                                                 test_labels, 2)
                if acc > exp_info['max_acc_bi']:
                    exp_info['max_acc_bi'] = acc
                    exp_info['max_acc_bi_params'] = param
                acc = amazon_experiments.pre_trained_experiments(combine_train_vectors, train_labels, combine_test_vectors,
                                                                 test_labels, 2)
                if acc > exp_info['max_acc_combine']:
                    exp_info['max_acc_combine'] = acc
                    exp_info['max_acc_combine_params'] = param
            elif dataset == 'imdb':
                acc = imdb_experiments.pre_trained_experiments(uni_train_vectors, train_labels, uni_test_vectors,
                                                               test_labels)
                if acc > exp_info['max_acc_uni']:
                    exp_info['max_acc_uni'] = acc
                    exp_info['max_acc_uni_params'] = param
                acc = imdb_experiments.pre_trained_experiments(bi_train_vectors, train_labels, bi_test_vectors,
                                                               test_labels)
                if acc > exp_info['max_acc_bi']:
                    exp_info['max_acc_bi'] = acc
                    exp_info['max_acc_bi_params'] = param
                acc = imdb_experiments.pre_trained_experiments(combine_train_vectors, train_labels, combine_test_vectors,
                                                               test_labels)
                if acc > exp_info['max_acc_combine']:
                    exp_info['max_acc_combine'] = acc
                    exp_info['max_acc_combine_params'] = param

            # Dump the info
            with open('./experiments/{}/accs.pkl'.format(dataset), 'w') as f:
                cPickle.dump(exp_info, f)

            e += 1

    return exp_info

if __name__ == '__main__':

    hyper_params = {
        'dim': [300, 600],
        'decay-c': [0.1, 0.],
        'grad-clip': [5., 8.],
        'maxlen-w': [20, 30, 50]
    }

    exp_info = run_grid_search(hyper_params, 'imdb')
    with open('./experiments/imdb/accs.pkl', 'w') as f:
        cPickle.dump(exp_info, f)

    exp_info = run_grid_search(hyper_params, 'amazon')
    with open('./experiments/amazon/accs.pkl', 'w') as f:
        cPickle.dump(exp_info, f)