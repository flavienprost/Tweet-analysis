import csv
import pickle

N_sentence=35000 #number of sentence in the training set to consider
N_jump=1580000/N_sentence #We take only one sentence every N_jump

def create_training_labels():
    """
    Creates cPickle file with a list containing the labels of the training data
    """

    print("Creating training labels..")
    sep = b","
    train =file("../Data/lab_train.dat", "r")

    labels = []
    i=0
    k=0
    for row in train:

        if i==N_sentence:break
        if k%N_jump==0: #Take only the sentence every N_jump
            label = row[0]
            labels.append(int(label))
            i=i+1

        k=k+1

    print('Number of lables equal to...')
    print(len(labels))
    pickle.dump(labels, open("training_labels_N_sentence="+ str(N_sentence) + ".p", "wb"))

create_training_labels()