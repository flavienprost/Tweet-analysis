from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from math import log
import csv
import cPickle as pickle
from sklearn.svm import LinearSVC
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
import numpy as np

stemmer = SnowballStemmer("english", ignore_stopwords=True)


def good_token(tok):
    """
    Returns False if degenerate token True otherwise
    """
    if len(tok) == 0 or tok[0] == '@' or tok[0:4] == 'http' or tok[0:3] == 'www':
        return False
    else:
        return True

punctuation = (',', '!', '?', '.')
negation = ("n't", 'not', 'never')


def desemphasyze(word):
    """
    Returns 2 occurences instead of 3+ for a word
    """

    l = len(word)
    if l < 3:
        return word
    else:
        result = ''
        for i in range(len(word) - 2):
            if word[i] == word[i + 1] and word[i+1] == word[i+2]:
                pass
            else:
                result += word[i]
        return result + word[-2:]

    if word[0] == '#':
        word.pop(0)


def process(strng):
    res = [ch for ch in strng] #if ch not in punctuation]
    sentence = ''.join(res)
    tokens = word_tokenize(sentence)
    return [stemmer.stem(unicode(tok)) for tok in tokens if good_token(tok)]


def labelize(tok, score):

    if score < 0.2:
        return 1
    elif score < 0.4:
        return 2
    elif score < 0.6:
        return 3
    elif score < 0.8:
        return 4
    else:
        return 5


def create_train_vectors():
    """
    Reads the training data and create the dictionary tok:occurence
    accordingly to the process function
    """
    print("Loading dictionary")

    dictio_pos = pickle.load(open("../data/dictio_pos.p", "rb"))
    dictio_neg = pickle.load(open("../data/dictio_neg.p", "rb"))

    train = open("../data/vec_train.dat", "r")
    train_labels = open("../data/labels_train.dat", "r")

    train_vectors = []
    train_labels = []

    for row in train:

        if count % 100000 == 0:
            print count
        count += 1

        tokens = process(row)
        feat = []

        for tok in tokens:

            try:
                score_p = dictio_pos[tok]*1.0
            except KeyError:
                score_p = 0

            try:
                score_n = dictio_neg[tok]*1.0
            except KeyError:
                score_n = 0

            s = score_n + score_p
            if s < 3:
                feat.append(3)
            else:
                feat.append(labelize(tok, score_p*1.0/s))

        if len(feat) > 20:
            feat = feat[:20]

        while len(feat) < 20:
            feat.append(0)

        train_vectors.append(feat)

    for row in train_labels:

        train_labels.append(int(row))

    train_vectors = np.array([np.array(elt) for elt in train_vectors])
    train_labels = np.array(train_labels)

    pickle.dump(train_vectors, open("../data/train_vectors.p", "wb"))
    pickle.dump(train_labels, open("../data/train_labels.p", "wb"))


def create_test_vectors():
    """
    Reads the training data and create the dictionary tok:occurence
    accordingly to the process function
    """
    print("Loading dictionary")

    dictio_pos = pickle.load(open("../data/dictio_pos.p", "rb"))
    dictio_neg = pickle.load(open("../data/dictio_neg.p", "rb"))

    sep = b","
    test = csv.reader(open("../data/testingdata.csv", "r"), delimiter=sep)

    test_vectors = []
    test_labels = []

    for row in test:

        if int(row[0]) == 2:
            pass
        else:
            if int(row[0]) == 0:
                lab = -1
            elif int(row[0]) == 4:
                lab = 1

            sentence = row[5].decode('utf-8', errors='ignore')
            tokens = process(sentence)
            feat = []

            for tok in tokens:

                try:
                    score_p = dictio_pos[tok]*1.0
                except KeyError:
                    score_p = 0

                try:
                    score_n = dictio_neg[tok]*1.0
                except KeyError:
                    score_n = 0

                s = score_n + score_p
                if s < 3:
                    feat.append(3)
                else:
                    feat.append(labelize(tok, score_p*1.0/s))

            if len(feat) > 20:
                feat = feat[:20]

            while len(feat) < 20:
                feat.append(0)

            test_vectors.append(feat)
            test_labels.append(lab)

    test_vectors = np.array([np.array(elt) for elt in test_vectors])
    test_labels = np.array(test_labels)

    pickle.dump(test_vectors, open("../data/test_vectors.p", "wb"))
    pickle.dump(test_labels, open("../data/test_labels.p", "wb"))

if __name__ == "__main__":
    create_train_vectors()
    create_test_vectors()
