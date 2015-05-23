from preprocess import *
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from math import log
import csv
import cPickle as pickle
from sklearn.svm import LinearSVC
from nltk.stem.porter import *
import itertools

stemmer = PorterStemmer()


def process(strng):

    tokens = word_tokenize(strng)
    #negatize(tokens)
    return [stemmer.stem(unicode(tok)) for tok in tokens if good_token(tok)]


def create_dictionaries():
    """
    Reads the training data and create the dictionary tok:occurence
    accordingly to the process function
    """
    print("Creating dictionary..")

    dictio_pos = {}
    dictio_neg = {}
    train = open('../data/vec_train.dat', 'r')
    labels =  open('../data/lab_train.dat', 'r')

    count = 0
    for sentence, lab in itertools.izip(train, labels):

        if count % 100000 == 0:
            print count
        count += 1

        tokens = process(sentence)
        if int(lab) == 1:
            for tok in tokens:
                if tok in dictio_pos:
                    dictio_pos[tok] += 1.
                else:
                    dictio_pos[tok] = 1.
        else:
            for tok in tokens:
                if tok in dictio_neg:
                    dictio_neg[tok] += 1.
                else:
                    dictio_neg[tok] = 1.

    # removing rare words
    new_dictio_pos = {key: dictio_pos[key] for key in dictio_pos if dictio_pos[key]>1}
    new_dictio_neg = {key: dictio_neg[key] for key in dictio_neg if dictio_neg[key]>1}

    #return dictio_pos, dictio_neg
    return new_dictio_pos, new_dictio_neg


def occurences2proba(dictio):
    total = sum(dictio.values())
    return {key: log(dictio[key]/total) for key in dictio}


if __name__ == "__main__":

    dictio_pos, dictio_neg = create_dictionaries()

    pickle.dump(dictio_pos, open("../data/dictio_pos.p", "wb"))
    pickle.dump(dictio_neg, open("../data/dictio_neg.p", "wb"))

    log_proba_pos = occurences2proba(dictio_pos)
    log_proba_neg = occurences2proba(dictio_neg)

    pickle.dump(log_proba_pos, open("../data/log_proba_pos.p", "wb"))
    pickle.dump(log_proba_neg, open("../data/log_proba_neg.p", "wb"))
