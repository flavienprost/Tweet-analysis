import pickle
import os.path
import numpy as np
import time


#INPUT
N_sentence=35000 #Number of training sentences
N_feat=8000 #Number of features



#STEP 0: TRAIN THE MODEL
#Loading the training data
t0=time.clock()
print('Loading the data')
current_dir= os.path.abspath(os.path.dirname(__file__))
parent_dir=os.path.dirname(current_dir)
training_data= pickle.load( open( parent_dir + '/Create_feature_label/training_features_sparse_N_feat='+ str(N_feat)+ '_N_sentences='+str(N_sentence) +'.p', "rb" ) )
print('Data loaded')

t1=time.clock()
print('Matrix to dense....')
training_data=training_data.todense()
print('Matrix to dense: DONE')

training_labels= pickle.load( open( parent_dir + '/Create_feature_label/training_labels_N_sentence='+ str(N_sentence) +'.p', "rb" ) )

indices_0=[]
indices_1=[]


for i in range(0,len(training_labels)):
	current_label='indices_'+ str(training_labels[i])
	exec("%s.append(%i)" % (current_label,i))




#Random Forest
from sklearn.ensemble import AdaBoostClassifier

print("Fitting the Adaboost Classifier")
t2=time.clock()
adab_clf = AdaBoostClassifier(n_estimators=35)
adab_clf= adab_clf.fit(training_data, training_labels)

#Compute the training error
predicted_train=adab_clf.predict(training_data)
train_error=np.mean(training_labels != predicted_train)
print("The Adaboost training error is %s" %(train_error))


del training_data
del training_labels
#Store the train predictions (stacking)
#print('Saving the training predictions')
#pickle.dump(predicted_train, open(parent_dir +"/Stacking step/Predictions_Classifier/Ada_train_predictions_N_feat="+ str(N_feat) + "_N_sentences="+ str(N_sentence) + ".p", "wb"))


#STEP 1:
# RUN THE MODEL ON THE VALIDATION SET
val_data= pickle.load( open( parent_dir + '/Create_feature_label/val_features_N_feat='+ str(N_feat)+ '.p', "rb" ) )
val_data=val_data.todense()

val_labels= pickle.load( open( parent_dir + '/Create_feature_label/val_labels.p', "rb" ) )

predicted_val=adab_clf.predict(val_data)
val_error=np.mean(val_labels != predicted_val)
print("The Adaboost validation error is %s" %(val_error))

print('Saving the labels')
pickle.dump(predicted_val, open(parent_dir +"/Stacking step/Predictions_Classifier/Ada_predictions_val_N_feat="+ str(N_feat) + "_N_sentences="+ str(N_sentence) + ".p", "wb"))


print('Predicting probabilites')
predicted_val_proba=adab_clf.predict_proba(val_data)
pickle.dump(predicted_val_proba, open(parent_dir +"/Stacking step/Predictions_Classifier/Ada_proba_val_N_feat="+ str(N_feat) + "_N_sentences="+ str(N_sentence) + ".p", "wb"))


#STEP 2:
#RUN THE MODEL ON THE TEST SET
print('Evaluating the test Error')
#Compute the testing error
testing_data= pickle.load( open( parent_dir + '/Create_feature_label/testing_features_N_feat='+ str(N_feat)+ '.p', "rb" ) )
testing_labels= pickle.load( open( parent_dir + '/Create_feature_label/testing_labels.p', "rb" ) )

indices_test_0=[]
indices_test_1=[]


for i in range(0,len(testing_labels)):
	current_label='indices_test_'+ str(testing_labels[i])
	exec("%s.append(%i)" % (current_label,i))

indices_test_0_1=indices_test_0+indices_test_1

testing_data=[testing_data[x] for x in indices_test_0_1]
testing_labels=[testing_labels[x] for x in indices_test_0_1]

predicted_test=adab_clf.predict(testing_data)
test_error=np.mean(testing_labels != predicted_test)
print("The Adaboost testing error is %s" %(test_error))

print('Saving the labels')
pickle.dump(predicted_test, open(parent_dir +"/Stacking step/Predictions_Classifier/Ada_predictions_test_N_feat="+ str(N_feat) + "_N_sentences="+ str(N_sentence) + ".p", "wb"))


print('Predicting probabilites')

#predicted_train_proba=adab_clf.predict_proba(training_data)
predicted_test_proba=adab_clf.predict_proba(testing_data)

print('Saving probabilities')
#pickle.dump(predicted_train_proba, open(parent_dir +"/Stacking step/Predictions_Classifier/Ada_proba_train_N_feat="+ str(N_feat) + "_N_sentences="+ str(N_sentence) + ".p", "wb"))
pickle.dump(predicted_test_proba, open(parent_dir +"/Stacking step/Predictions_Classifier/Ada_proba_test_N_feat="+ str(N_feat) + "_N_sentences="+ str(N_sentence) + ".p", "wb"))


t3=time.clock()

print(t1-t0)
print(t2-t1)
print(t3-t2)
