################# Advanced Machine Learning #################

Twitter Sentiment Analysis using Blending on various Embedding-Based Classifiers

Robert Dadashi-Tazehozi rd2669
Flavien Prost fp2316

####################################################################


0. Getting Data

The data can be dowloaded with the following link:
http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip

The training data and testing data have to be renamed: trainingdata.csv 
and testingdata.csv

1. Preprocess

The preprocessing step is done by compiling the preprocess.py script in the 
preprocess folder. It creates 6 .dat file corresponding to the training, validating and
testing sentences and labels.

2. Naive Bayes 

Naive Bayes method requires to create the dictionaries thanks to the create_dictionaries.py,
and chose a set of best features using Select_best_features.py.
Then the naives bayes method can be compiled using naive_bayes.py

3. Doc2Vec

The Doc2Vec script enables to create the doc2vec model, it requires downloading the newest 
version of gensim via https://github.com/piskvorky/gensim/
The model is computed using create_model_w2v.py
The vectors are stored in a pickle file, and we can then used classifiers on these vectors. 

4. P-grained method

The P-grained method requires creating the vectors using create_k_vectors.py. The classifier
are then coded in log_reg.py, mlp.py, (where Theano is required), k_vectors_classifiers
(where scikit learn is required).

5. Feature selection + Random Forest + Adaboost
2 steps: 
i- Select the best features and create the corresponding features sets: all these files are in the folder "Create_feature_label". It will create pickle files in the same folder: one pickle for every type (test, train or vaildation) and for every choice of parameter (n_sentence, n_features)

ii- Use classifiers on the features: in the folder "Classifier_feature_selection"
After step i, you can run any classifier and you can change the parameters (N_features,N_sentence).
This step will create containing the predictions.


6. Blending 
In the folder "Stacking Step"
The predictions of any classifier seen before will go in the folder "Predictions"
Then 3 different possibilities: 
-Stacking with probabilities
-Stacking with predictions
-Majority voting
