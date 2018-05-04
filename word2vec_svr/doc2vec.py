from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.word2vec import Word2Vec
from sklearn.metrics import mean_squared_error, f1_score
from sklearn.svm import SVR
import data_processor
import numpy as np
import gensim
import pickle


def train_word2vec(args):
    X, Y = data_processor.read_data(args.train_path) X_t, Y_t = data_processor.read_data(args.test_path)

    documents = np.concatenate((X, X_t), axis=0)
    model = Word2Vec(documents, size=300, window=8, min_count=0, workers=4)
    model.save('model/word2vec')


def train_doc2vec(args):
    X, Y = data_processor.read_data(args.train_path)
    X_t, Y_t = data_processor.read_data(args.test_path)

    documents = [TaggedDocument(x, f'{idx}') for idx, x in enumerate(np.concatenate((X, X_t), axis=0))]
    model = Doc2Vec(documents, vector_size=300, window=8, min_count=0, workers=4)
    model.save('model/doc2vec')


def sentiment_toclass(Y):
    Y_c = []
    for y in Y:
        if y > 0:
            Y_c.append(0)
        if y == 0:
            Y_c.append(1)
        if y < 0:
            Y_c.append(2)
    return np.array(Y_c)


def train_svr(X, Y, X_t, Y_t):
    clf = SVR(C=1, epsilon=0.2)
    clf.fit(X, Y)
    pickle.dump(clf, open('model/clf', 'wb'))

    Y_pred = clf.predict(X_t)
    print(f'mse:{mean_squared_error(Y_t, Y_pred)}')
    print(f'macro:{f1_score(sentiment_toclass(Y_t), sentiment_toclass(Y_pred), average="macro")}')
    print(f'micro:{f1_score(sentiment_toclass(Y_t), sentiment_toclass(Y_pred), average="micro")}')


def infer_doc2vec(args):
    X, Y = data_processor.doc2vec_feature(args.train_path)
    X_t, Y_t = data_processor.doc2vec_feature(args.test_path)
    train_svr(X, Y, X_t, Y_t)


def infer_word2vec(args):
    X, Y = data_processor.word2vec_feature_max(args.train_path)
    X_t, Y_t = data_processor.word2vec_feature_max(args.test_path)
    train_svr(X, Y, X_t, Y_t)


def main(args):
    np.random.seed(1126)

    infer_doc2vec(args)
    #infer_word2vec(args)
    #train_word2vec(args)
    #train_doc2vec(args)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='../training_set.json')
    parser.add_argument('--test_path', default='../test_set.json')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(parse_args())
