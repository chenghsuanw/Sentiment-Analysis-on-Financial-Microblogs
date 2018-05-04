import numpy as np
import pandas
import gensim
import json


def word2vec_feature_min(file_name):
    model = gensim.models.word2vec.Word2Vec.load('model/word2vec')
    X = []
    Y = []
    data = json.load(open(file_name))
    for tw in data:
        if type(tw['snippet']) == list: 
            tok = ' '.join([x.upper() for x in tw['snippet']]).split(' ')
        else:
            tok = tw['snippet'].upper().split(' ')

        maxv = model.wv[tok[0]]
        for word in tok:
            if sum(maxv) > sum(model.wv[word]):
                maxv = model.wv[word]

        X.append(maxv)
        Y.append(float(tw['sentiment']))
    return np.array(X), np.array(Y)


def word2vec_feature_max(file_name):
    model = gensim.models.word2vec.Word2Vec.load('model/word2vec')
    X = []
    Y = []
    data = json.load(open(file_name))
    for tw in data:
        if type(tw['snippet']) == list: 
            tok = ' '.join([x.upper() for x in tw['snippet']]).split(' ')
        else:
            tok = tw['snippet'].upper().split(' ')

        maxv = model.wv[tok[0]]
        for word in tok:
            if sum(maxv) < sum(model.wv[word]):
                maxv = model.wv[word]

        X.append(maxv)
        Y.append(float(tw['sentiment']))
    return np.array(X), np.array(Y)


def word2vec_feature_mean(file_name):
    model = gensim.models.word2vec.Word2Vec.load('model/word2vec')
    X = []
    Y = []
    data = json.load(open(file_name))
    for tw in data:
        if type(tw['snippet']) == list: 
            tok = ' '.join([x.upper() for x in tw['snippet']]).split(' ')
        else:
            tok = tw['snippet'].upper().split(' ')

        tmp = np.zeros(300)
        for word in tok:
            tmp += model.wv[word]
        tmp /= len(tok)
        X.append(tmp)
        Y.append(float(tw['sentiment']))
    return np.array(X), np.array(Y)


def doc2vec_feature(file_name):
    model = gensim.models.word2vec.Word2Vec.load('model/doc2vec')
    X = []
    Y = []
    data = json.load(open(file_name))
    for tw in data:
        if type(tw['snippet']) == list: 
            tok = ' '.join([x.upper() for x in tw['snippet']]).split(' ')
        else:
            tok = tw['snippet'].upper().split(' ')
        X.append(model.infer_vector(tok))
        Y.append(float(tw['sentiment']))
    return np.array(X), np.array(Y)


def read_data(file_name):
    X = []
    Y = []
    data = json.load(open(file_name))
    for tw in data:
        if type(tw['snippet']) == list: 
            X.append(' '.join([x.upper() for x in tw['snippet']]).split(' '))
        else:
            X.append(tw['snippet'].upper().split(' '))
        Y.append(float(tw['sentiment']))
    return np.array(X), np.array(Y)
