#Program which takes selected_features.py as an input and builds the training features


#INPUT
N_sentence=35000#Number of training sentences we want
N_features=8000


N_jump=1580000/N_sentence #We take only one sentence every N_jump


import cPickle as pickle
from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix
import time

t0=time.clock()
selected_features= pickle.load( open( 'Selected_features_N_feat=' +str(N_features) +'.p', "rb" ) )
N_features=len(selected_features)
print('N_features=%s' %N_features)

print(selected_features[1])
#Import the training data pre-processed
train_sentence=file("../Data/vec_train.dat","r")



#STEP 1:Pre-process the features
# creates a dictionary which gives the position in Selected_features for every feature
t1=time.clock()
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


#STEP 2: Create the features
features=[]
t2=time.clock()

print('Starting to create the features array')


for i in range(0,N_sentence):

    current_feature=np.zeros(shape=(N_features), dtype=int)
    
    
    sentence=train_sentence.readline()

    sentence=sentence.strip()
    sentence=sentence.split(" ")

    for word in sentence:
        if is_feature(word):
            current_feature[position_feat[word]]=current_feature[position_feat[word]]+1

    features.append(current_feature)

    for k in range(1, N_jump):
        sentence=train_sentence.readline()





#Put the matrix into a sparse matrix
t3=time.clock()
features_sparse=csr_matrix(features)
print('Saving the features into a pickle file')
pickle.dump(features_sparse, open("training_features_sparse_N_feat="+ str(N_features) + "_N_sentences="+ str(N_sentence) + ".p", "wb"))
t4=time.clock()


print('t1-t0')
print(t1-t0)
print(t2-t1)
print(t3-t2)
print(t4-t3)


