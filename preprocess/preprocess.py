import csv
import cPickle as pickle
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english", ignore_stopwords=True)


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
def create_datasets():
    """
    Converts training data csvfile with sentences into cleaned sentences
    """

    print("Converting training and validating data..")
    train_file = open('../data/vec_train.dat', 'w')
    train_lab = open('../data/lab_train.dat', 'w')
    validate_file = open('../data/vec_validate.dat', 'w')
    validate_lab = open('../data/lab_validate.dat', 'w')

    sep = b","
    train = csv.reader(open("../data/trainingdata.csv", "r"), delimiter=sep)

    count = -1
    for row in train:
        count += 1
        s = row[5].decode('utf-8', errors='ignore')
        lign = s.lower()
        lign = lign.encode('ascii', 'ignore')
        s = ''.join(ch for ch in lign if ch not in punctuation)
        S = s.split(' ')
        new = [stemmer.stem(desemphasyze(tok)) for tok in S if good_token(tok)]
        newnew = ' '.join(new)

        label = str(int(row[0])/4)

        if count < 790000:
            train_file.write('%s\n' % newnew)
            train_lab.write('%s\n' % label)
        elif count < 800000:
            validate_file.write('%s\n' % newnew)
            validate_lab.write('%s\n' % label)
        elif count < 1590000:
            train_file.write('%s\n' % newnew)
            train_lab.write('%s\n' % label)
        else:
            validate_file.write('%s\n' % newnew)
            validate_lab.write('%s\n' % label)

    train_file.close()
    validate_file.close()

    print("Converting testing data..")
    test_file = open('../data/vec_test.dat', 'w')
    test_lab = open('../data/lab_test.dat', 'w')
    sep = b","
    test = csv.reader(open("../data/testingdata.csv", "r"), delimiter=sep)


    for row in test:
        if row[0] == '2':
            pass
        else:
            s = row[5].decode('utf-8', errors='ignore')
            lign = s.lower()
            lign = lign.encode('ascii', 'ignore')
            s = ''.join(ch for ch in lign if ch not in punctuation)
            S = s.split(' ')
            new = [stemmer.stem(desemphasyze(tok)) for tok in S if good_token(tok)]
            newnew = ' '.join(new)
            test_file.write('%s\n' % newnew)

            label = str(int(row[0])/4)
            test_lab.write('%s\n' % label)

    test_file.close()
    test_lab.close()

    print("Creating testing labels..")
    sep = b","
    test = csv.reader(open("../data/testingdata.csv", "r"), delimiter=sep)

    labels = []
    for row in test:
        label = row[0]
        labels.append(int(label))

    pickle.dump(labels, open("../data/testing_labels.p", "wb"))

if __name__ == "__main__":
    create_datasets()
