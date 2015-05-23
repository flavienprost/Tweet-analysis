import pickle
import os
from collections import defaultdict
import operator

#Parameters
N=10000 #Number of selected features we want to keep
Type='Gini' #Type of criteria selection we want to use
min_occurrences=30 #the minimum occurences the selected features should have




#Import the labels
training_labels= pickle.load( open('training_labels_all_corpus.p', "rb" ) )

#Import the training data pre-processed
train_sentence=file("../Data/vec_train.dat","r")



#STEP 1:
# Create the one dictionary for each class which counts the number of occurrences for every word
# And one dictionary which combines the two classes

dictionary=defaultdict()
dict_4=defaultdict()
dict_0=defaultdict()

sentence=train_sentence.readline()
n_sentence=0

number_words_0=0 # total number of words in sentences labeled 0
number_words_4=0 # total number of words in sentences labeled 4


while sentence:
    n_sentence=n_sentence+1

    sentence=sentence.strip()
    sentence=sentence.split(" ")

    label=training_labels[n_sentence-1]

    for word in sentence:
        try:
            dictionary[word]==0
        except:
            dictionary[word]=0
        dictionary[word]=dictionary[word]+1

        if label==0:
            try:
                dict_0[word]==0
            except:
                dict_0[word]=0
            dict_0[word]=dict_0[word]+1
            number_words_0+=1
        else:
            try:
                dict_4[word]==0
            except:
                dict_4[word]=0
            dict_4[word]=dict_4[word]+1
            number_words_4+=1

    sentence=train_sentence.readline()

train_sentence.close()

print(len(dictionary))
print(len(dict_0))
print(len(dict_4))



#STEP 2:
#Select the words:

#STEP 2-a:
# Create a dictionary called difference
# It evaluates the difference of probability of occurrences in the two classes for every words

difference=defaultdict(float)
proba_0=defaultdict(float)
proba_4=defaultdict(float)

for word in dictionary:

    try:
        dict_0[word]==0.
    except:
        dict_0[word]=1 #We set it equal to 0 to be sure that the proba is not 0

    try:
        dict_4[word]==0.
    except:
        dict_4[word]=1 #We set it equal to 0 to be sure that the proba is not 0

    if dictionary[word]>min_occurrences:
        proba_0[word]=float(dict_0[word])/float(dictionary[word])
        proba_4[word]=float(dict_4[word])/float(dictionary[word])
        difference[word]=(proba_0[word]*(1.-proba_0[word])+proba_4[word]*(1.-proba_4[word]))

    else:
        difference[word]=1

#print(difference['i'])
#print(len(difference))

#STEP 2-b:
#Sort the dictionary with the values

sorted_difference = sorted(difference.items(), key=operator.itemgetter(1)) #in increasing order



#STEP 2-c
#store the best features
features=[]

for k in range (0,N):
    features.append(sorted_difference[k][0])
print(len(features))
print('Saving the features into pickle')
pickle.dump(features, open("Selected_features_N_feat=" + str(N) + ".p", "wb"))



#Print the 100 best features
print('We print the best features and their Gini information')
for i in range(0,100):
    print(sorted_difference[i])
    print(difference[sorted_difference[i][1]])







