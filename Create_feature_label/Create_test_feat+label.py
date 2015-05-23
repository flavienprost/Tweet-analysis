#Program which takes selected_features.py as an input and builds the testing features

#INPUT
N_feat=12000


import pickle
from collections import defaultdict
import numpy as np
import csv

selected_features= pickle.load( open('Selected_features_N_feat='+ str(N_feat) + '.p', "rb" ) )
N_features=len(selected_features)
print('N_features=%s' %N_features)

#Import the training data pre-processed
test_sentence=file("../Data/vec_test.dat","r")



#STEP 1:Pre-process the features
# creates a dictionary which gives the position in Selected_features for every feature

position_feat=defaultdict()

for i in range(0,len(selected_features)):
    position_feat[selected_features[i]]=i

#Creates a function which tells whether a word is part of the selected features
def is_feature(word):
    is_selected=True
    try:
        position_feat[word]==0

    except:
        is_selected=False
    return(is_selected)


#STEP 2: Create the  test features
features=[]

print('Starting to create the features array')

sentence=test_sentence.readline()
k=0
while sentence and k<498:
    k=k+1
    current_feature=np.zeros(shape=(N_features), dtype=int)

    sentence=sentence.strip()
    sentence=sentence.split(" ")

    for word in sentence:
        if is_feature(word):
            current_feature[position_feat[word]]=current_feature[position_feat[word]]+1

    features.append(current_feature)
    sentence=test_sentence.readline()
test_sentence.close()


pickle.dump(features, open("testing_features_N_feat="+ str(N_features) +".p", "wb"))

#STEP 3: Create test_labels

def create_training_labels():
    """
    Creates cPickle file with a list containing the labels of the training data
    """

    print("Creating training labels..")
    sep = b","
    test = file("../Data/lab_test.dat", "r")

    labels = []
    i=0
    for row in test:
        i=i+1
        label = row[0]
        labels.append(int(label))

    pickle.dump(labels, open("testing_labels.p", "wb"))

create_training_labels()