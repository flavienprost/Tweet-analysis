import os
import nltk
import csv
import cPickle as pickle
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import LabeledLineSentence
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english", ignore_stopwords=True)
# nombre de donnees : 16000000 / coeff
coeff = 1


def good_token(tok):
    """
    Returns False if degenerate token True otherwise
    """
    if len(tok) == 0 or tok[0] == '@' or tok[0:4] == 'http' or tok[0:3] == 'www':
        return False
    else:
        return True


def desemphasyze(word):
    """
    Returns 2 occurences instead of 3+ for a word
    """

    if word[0] == '#':
        word = word[1:]

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

punctuation = (',', '?', '.')


#Create the training_data
def convert_train():
    """
    Converts training data csvfile with sentences into cleaned sentences
    """
    print("Converting training data..")
    train_file = open('../data/vec_train.dat', 'w')
    sep = b","
    train = csv.reader(open("../data/trainingdata.csv", "r"), delimiter=sep)

    count = -1
    for row in train:
        count += 1
        if count % coeff == 0:
            s = row[5].decode('utf-8', errors='ignore')
            lign = s.lower()
            lign = lign.encode('ascii', 'ignore')
            s = ''.join(ch for ch in lign if ch not in punctuation)
            S = s.split(' ')
            new = [stemmer.stem(desemphasyze(tok)) for tok in S if good_token(tok)]
            newnew = ' '.join(new)
            train_file.write('%s\n' % newnew)
        else:
            continue

    train_file.close()


#Create the testing_data
def convert_test():
    """
    Converts testing data csvfile with sentences into cleaned sentences
    """
    print("Converting testing data..")
    test_file = open('../data/vec_test.dat', 'w')
    sep = b","
    test = csv.reader(open("../data/testingdata.csv", "r"), delimiter=sep)


    for row in test:
        s = row[5].decode('utf-8', errors='ignore')
        lign = s.lower()
        lign = lign.encode('ascii', 'ignore')
        s = ''.join(ch for ch in lign if ch not in punctuation)
        S = s.split(' ')
        new = [stemmer.stem(desemphasyze(tok)) for tok in S if good_token(tok)]
        newnew = ' '.join(new)
        test_file.write('%s\n' % newnew)

    test_file.close()


def create_testing_labels():
    """
    Creates cPickle file with a list containing the labels of the testing data
    """

    print("Creating testing labels..")
    sep = b","
    test = csv.reader(open("../data/testingdata.csv", "r"), delimiter=sep)

    labels = []
    for row in test:
        label = row[0]
        labels.append(int(label))

    pickle.dump(labels, open("../data/testing_labels.p", "wb"))


def visualize():
    train_file = open('../data/vec_train.dat', 'r')

    i = 1
    while i < 30:
        l = train_file.readline()
        print(l)
        i = i + 1


def create_training(model):

    print("Creating training dataset..")
    sep = b","
    train = csv.reader(open("../data/trainingdata.csv", "r"), delimiter=sep)

    labels = []
    features = []
    count = 0
    id_sent = -1
    corres = {}
    for row in train:
        id_sent += 1
        if id_sent % coeff == 0:
            try:
                s = "SENT_" + str(id_sent/coeff)
                features.append(model[s])
                label = row[0]
                if label == 2:
                    print label
                labels.append(int(label))
                corres[count] = id_sent
                count += 1
            except KeyError:
                continue
        else:
            continue

    pickle.dump(labels, open("../data/training_labels.p", "wb"))
    pickle.dump(features, open("../data/training_features.p", "wb"))
    pickle.dump(corres, open("../data/correspondance.p", "wb"))


def create_testing_features(model):

    print("Creating testing features from doc2vec model..")
    features = []
    f = open('../data/vec_test.dat', 'r')
    for row in f:
        features.append(model.infer_vector(row))

    pickle.dump(features, open("../data/testing_features.p", "wb"))

if __name__ == "__main__":
    convert_train()
    convert_test()
    #print('Building model..')
    #sentences = LabeledLineSentence("../data/vec_train.dat")
    #trained_model = Doc2Vec(sentences, size=400, alpha=0.025, window=8, min_count=5, sample=0, seed=1, workers=1, min_alpha=0.0001, dm=1, hs=1, negative=0, dm_mean=0, train_words=True, train_lbls=True)
    #trained_model.save("doc2vec_model")
    #trained_model = Doc2Vec.load("doc2vec_model")
    #create_training(trained_model)
    #create_testing_labels()
    #create_testing_features(trained_model)
