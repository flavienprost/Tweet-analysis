import cPickle as pickle
from nltk.stem.snowball import SnowballStemmer
import numpy as np
from preprocess import *
from create_dictionaries import *
from create_vectors import *
import itertools

stemmer = SnowballStemmer("english", ignore_stopwords=True)


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


def create_vectors(strng, coeff, v_size):
    """
    Reads the training data and create the dictionary tok:occurence
    accordingly to the process function
    """
    print("Loading dictionary")

    dictio_pos = pickle.load(open("../data/dictio_pos.p", "rb"))
    dictio_neg = pickle.load(open("../data/dictio_neg.p", "rb"))

    train = open("../data/vec_" + strng + ".dat", "r")
    labels = open("../data/lab_" + strng + ".dat", "r")

    train_vectors = []
    train_labels = []
    count = 0

    print("Creating vectors")

    for sentence, lab in itertools.izip(train,labels):

        if count % 100000 == 0:
            print count
        count += 1

        tokens = process(sentence)
        feat = []

        if count % coeff == 0:
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

            if len(feat) > v_size:
                feat = feat[:v_size]

            while len(feat) < v_size:
                feat.append(0)

            train_vectors.append(feat)
            train_labels.append(int(lab))

    train_vectors = np.array([np.array(elt) for elt in train_vectors])
    train_labels = np.array(train_labels)

    pickle.dump(train_vectors, open("../data/" + strng + "_k_vectors.p", "wb"))
    pickle.dump(train_labels, open("../data/" + strng + "_k_labels.p", "wb"))


if __name__ == "__main__":
    create_vectors("train", coeff=80, v_size=20)
    create_vectors("validate", coeff=1, v_size=20)
    create_vectors("test", coeff=1, v_size=20)
