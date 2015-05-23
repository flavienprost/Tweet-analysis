import pickle
import os.path
import numpy as np


#INPUT
N_sentence=50000 #Number of training sentences
N_feat=10000 #Number of features



#STEP 1: TRAIN THE MODEL

#Loading the training data
print('Loading the data')
current_dir= os.path.abspath(os.path.dirname(__file__))
parent_dir=os.path.dirname(current_dir)
training_data= pickle.load( open( parent_dir + '/Create_feature_label/training_features_sparse_N_feat='+ str(N_feat)+ '_N_sentences='+str(N_sentence) +'.p', "rb" ) )
training_labels= pickle.load( open( parent_dir + '/Create_feature_label/training_labels_N_sentence='+ str(N_sentence) +'.p', "rb" ) )
print('Data Loaded')

print('Matrix to dense....')
training_data=training_data.todense()
print('Matrix to dense: DONE')

indices_0=[]
indices_1=[]
#indices_4=[]

for i in range(0,len(training_labels)):
	current_label='indices_'+ str(training_labels[i])
	exec("%s.append(%i)" % (current_label,i))


#Random Forest
from sklearn.ensemble import RandomForestClassifier

print("Fitting the Random Forest Classifier")
rfc = RandomForestClassifier(n_estimators=50)
rfc = rfc.fit(training_data, training_labels)


#Compute the training error
#predictions_train=rfc.predict(training_data)
#train_error = np.mean(predictions_train != training_labels)
#print("The Random forest training error is %s" %(train_error))

#Store the train predictions (stacking)
#print('Saving the training predictions')
#pickle.dump(predictions_train, open(parent_dir +"/Stacking step/Predictions_Classifier/Rfc_train_predictions_N_feat="+ str(N_feat) + "_N_sentences="+ str(N_sentence) + ".p", "wb"))

del training_data
del training_labels

#STEP 2: PREDICT ON THE VALIDATION SET
val_data= pickle.load( open( parent_dir + '/Create_feature_label/val_features_N_feat='+ str(N_feat)+ '.p', "rb" ) )
val_data=val_data.todense()
val_labels= pickle.load( open( parent_dir + '/Create_feature_label/val_labels.p', "rb" ) )

predicted_val=rfc.predict(val_data)
val_error=np.mean(val_labels != predicted_val)
print("The Logistic regression validation error is %s" %(val_error))

print('Saving the labels')
pickle.dump(predicted_val, open(parent_dir +"/Stacking step/Predictions_Classifier/Rfc_predictions_val_N_feat="+ str(N_feat) + "_N_sentences="+ str(N_sentence) + ".p", "wb"))


print('Predicting probabilites')
predicted_val_proba=rfc.predict_proba(val_data)
pickle.dump(predicted_val_proba, open(parent_dir +"/Stacking step/Predictions_Classifier/Rfc_proba_val_N_feat="+ str(N_feat) + "_N_sentences="+ str(N_sentence) + ".p", "wb"))






#STEP 3:
#PREDICT ON THE TEST SET

print('Evaluating the test Error')

#Compute the testing error
testing_data= pickle.load( open( parent_dir + '/Create_feature_label/testing_features_N_feat='+ str(N_feat)+ '.p', "rb" ) )
testing_labels= pickle.load( open( parent_dir + '/Create_feature_label/testing_labels.p', "rb" ) )

indices_test_0=[]
indices_test_1=[]
#indices_test_4=[]


for i in range(0,len(testing_labels)):
	current_label='indices_test_'+ str(testing_labels[i])
	exec("%s.append(%i)" % (current_label,i))

indices_test_0_1=indices_test_0+indices_test_1

testing_data=[testing_data[x] for x in indices_test_0_1]
testing_labels=[testing_labels[x] for x in indices_test_0_1]

predictions=rfc.predict(testing_data)

test_error = np.mean(predictions != testing_labels)
print("The random forest testing error is %s" %(test_error))


print('Saving the predictions')
pickle.dump(predictions, open(parent_dir +"/Stacking step/Predictions_Classifier/Rfc_predictions_test_N_feat="+ str(N_feat) + "_N_sentences="+ str(N_sentence) + ".p", "wb"))


print('Predicting probabilites')
#predicted_train_proba=rfc.predict_proba(training_data)
#pickle.dump(predicted_train_proba, open(parent_dir +"/Stacking step/Predictions_Classifier/Rfc_proba_train_N_feat="+ str(N_feat) + "_N_sentences="+ str(N_sentence) + ".p", "wb"))

predicted_test_proba=rfc.predict_proba(testing_data)
pickle.dump(predicted_test_proba, open(parent_dir +"/Stacking step/Predictions_Classifier/Rfc_proba_test_N_feat="+ str(N_feat) + "_N_sentences="+ str(N_sentence) + ".p", "wb"))
