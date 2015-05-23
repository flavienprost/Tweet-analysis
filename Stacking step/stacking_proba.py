import pickle
import os.path
import numpy as np
import time
from sklearn import linear_model


N_feat=10000
N_sentence=50000
Classifier=['Log','Rfc']
Classifier2=['proba_naivebayes']

current_dir= os.path.abspath(os.path.dirname(__file__))
parent_dir=os.path.dirname(current_dir)
training_labels= pickle.load( open( parent_dir + '/Create_feature_label/val_labels.p', "rb" ) )

features_train_stacking=[]
classif_predictions=pickle.load(open(current_dir+ '/Predictions_Classifier/Log'+ "_proba_val_N_feat="+ str(N_feat) + "_N_sentences="+ str(N_sentence)+ ".p", "rb"))

print('Creating the table')
for i in range(0,20000):#length of the validation set
    features_train_stacking.append([classif_predictions[i][0]])

for clf in Classifier:
    if clf<>'Log':

        classif_predictions=pickle.load(open(current_dir+ '/Predictions_Classifier/'+clf+ "_proba_val_N_feat="+ str(N_feat) + "_N_sentences="+ str(N_sentence) + ".p", "rb"))

        for k in range(0,20000):
            features_train_stacking[k].append(classif_predictions[k][0])

for clf2 in Classifier2:

        classif_predictions=pickle.load(open(current_dir+ '/Predictions_Classifier/val_predict_'+clf2+'.p', "rb"))
        for k in range(0,20000):
            features_train_stacking[k].append(classif_predictions[k])




print(features_train_stacking[0])
print(len(features_train_stacking))

print("Fitting the Logistic Regression Classifier for stacking")
log_reg = linear_model.LogisticRegression(C=1)
log_model= log_reg.fit(features_train_stacking, training_labels)

#Compute the training error
predicted_train=log_model.predict(features_train_stacking)
train_error=np.mean(training_labels != predicted_train)
print("The Logistic Regression training error (stacking step) is %s" %(train_error))


#Compute the test error
testing_labels= pickle.load( open( parent_dir + '/Create_feature_label/testing_labels.p', "rb" ) )

indices_test_0=[]
indices_test_1=[]

for i in range(0,len(testing_labels)):
	current_label='indices_test_'+ str(testing_labels[i])
	exec("%s.append(%i)" % (current_label,i))

indices_test_0_1=indices_test_0+indices_test_1

testing_labels=[testing_labels[x] for x in indices_test_0_1]

features_test_stacking=[]
classif_predictions=pickle.load(open(current_dir+ '/Predictions_Classifier/'+Classifier[0]+ "_proba_test_N_feat="+ str(N_feat) + "_N_sentences="+ str(N_sentence)+ ".p", "rb"))

print('Creating the table')
for i in range(0,len(classif_predictions)):
    features_test_stacking.append([classif_predictions[i][0]])

for clf in Classifier:
    if clf<>Classifier[0]:
        classif_predictions=pickle.load(open(current_dir+ '/Predictions_Classifier/'+clf+ "_proba_test_N_feat="+ str(N_feat) + "_N_sentences="+ str(N_sentence) + ".p", "rb"))

        for k in range(0,len(classif_predictions)):
            features_test_stacking[k].append(classif_predictions[k][0])

for clf2 in Classifier2:
        classif_predictions=pickle.load(open(current_dir+ '/Predictions_Classifier/test_predict_'+clf2+'.p', "rb"))

        for k in range(0,len(classif_predictions)):
            features_test_stacking[k].append(classif_predictions[k])


test_predictions=log_model.predict(features_test_stacking)
test_error=np.mean(testing_labels!= test_predictions)

print("The Logistic Regression testing error (stacking step) is %s" %str(test_error))

