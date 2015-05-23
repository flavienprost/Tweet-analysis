from create_dictionaries import *
import itertools
import cPickle as pickle
import pickle as pic

N = 25000
best1_feat = pic.load(open('Selected_features_N_feat='+str(N)+'.p','r'))
best_feat = {elt: 0 for elt in best1_feat}

def classify(strng, log_proba_pos, log_proba_neg):

    tokens = process(strng)
    logp_pos = 0
    logp_neg = 0
    for tok in tokens:
        if tok in log_proba_pos and tok in best_feat:
            logp_pos += log_proba_pos[tok]
        if tok in log_proba_neg and tok in best_feat:
            logp_neg += log_proba_neg[tok]
    if logp_pos >= logp_neg:
        return 1
    else:
        return 0


def get_proba(strng, log_proba_pos, log_proba_neg):

    tokens = process(strng)
    logp_pos = 0.
    logp_neg = 0.
    for tok in tokens:
        if tok in log_proba_pos:
            logp_pos += log_proba_pos[tok]
        if tok in log_proba_neg:
            logp_neg += log_proba_neg[tok]
    if logp_pos + logp_neg == 0:
        return 0.5
    else:
        return logp_neg / (logp_pos + logp_neg)


def test_classification(log_proba_pos, log_proba_neg, strng):

    err = 0.
    print('Testing..')
    test = open("../data/vec_"+strng+".dat", "r")
    labels = open("../data/lab_"+strng+".dat", "r")
    count = 0

    #prediction_bin = []
    #prediction_proba = []
    for sentence, lab in itertools.izip(test, labels):

        if int(lab) != classify(sentence, log_proba_pos, log_proba_neg):
            err += 1.
        count += 1
        #prediction_bin.append(classify(sentence, log_proba_pos, log_proba_neg))
        #prediction_proba.append(get_proba(sentence, log_proba_pos, log_proba_neg))

    return err / count #, prediction_bin, prediction_proba

if __name__ == "__main__":

    log_proba_pos = pickle.load(open("../data/log_proba_pos.p", "rb"))
    log_proba_neg = pickle.load(open("../data/log_proba_neg.p", "rb"))

    score = test_classification(log_proba_pos, log_proba_neg, "test")

    #pickle.dump(prediction_bin, open('../data/test_predict_bin_naivebayes.p' ,'w'))
    #pickle.dump(prediction_proba, open('../data/test_predict_proba_naivebayes.p' ,'w'))

    print(('Testing error %f') % (score))

    score = test_classification(log_proba_pos, log_proba_neg, "validate")

    print(('Validating error %f') % (score))

    #pickle.dump(prediction_bin, open('../data/val_predict_bin_naivebayes.p' ,'w'))
    #pickle.dump(prediction_proba, open('../data/val_predict_proba_naivebayes.p' ,'w'))


