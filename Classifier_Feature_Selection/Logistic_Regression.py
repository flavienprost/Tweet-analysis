import pickle
import os.path
import numpy as np


#INPUT
N_sentence=50000 #Number of training sentences
N_feat=3000 #Number of features


#STEP 1: TRAIN THE MODEL
#Loading the training
print('Loading the data')
current_dir= os.path.abspath(os.path.dirname(__file__))
parent_dir=os.path.dirname(current_dir)
training_data= pickle.load( open( parent_dir + '/Create_feature_label/training_features_sparse_N_feat='+ str(N_feat)+ '_N_sentences='+str(N_sentence) +'.p', "rb" ) )
print('Data loaded')


print('Matrix to dense....')
training_data=training_data.todense()
print('Matrix to dense: DONE')


training_labels= pickle.load( open( parent_dir + '/Create_feature_label/training_labels_N_sentence='+ str(N_sentence) +'.p', "rb" ) )

indices_0=[]
indices_1=[]
#indices_4=[]

for i in range(0,len(training_labels)):
	current_label='indices_'+ str(training_labels[i])
	exec("%s.append(%i)" % (current_label,i))



#Random Forest
from sklearn import linear_model

print("Fitting the Logistic Regression Classifier")
log_reg = linear_model.LogisticRegression(C=1)
log_model= log_reg.fit(training_data, training_labels)


print('Model Trained')

#Compute the training error
#print('Computing train error')
#predicted_train=log_model.predict(training_data)
#train_error=np.mean(training_labels != predicted_train)
#print("The Logistic Regression training error is %s" %(train_error))

#del predicted_train
del training_data

#Store the train predictions (stacking)
#print('Saving the training predictions')
#pickle.dump(predicted_train, open(parent_dir +"/Stacking step/Predictions_Classifier/Log_train_predictions_N_feat="+ str(N_feat) + "_N_sentences="+ str(N_sentence) + ".p", "wb"))


#STEP 2: PREDICT ON THE VALIDATION SET
val_data= pickle.load( open( parent_dir + '/Create_feature_label/val_features_N_feat='+ str(N_feat)+ '.p', "rb" ) )
val_data=val_data.todense()
val_labels= pickle.load( open( parent_dir + '/Create_feature_label/val_labels.p', "rb" ) )

predicted_val=log_model.predict(val_data)
val_error=np.mean(val_labels != predicted_val)
print("The Logistic regression validation error is %s" %(val_error))



print('Saving the labels')
pickle.dump(predicted_val, open(parent_dir +"/Stacking step/Predictions_Classifier/Log_predictions_val_N_feat="+ str(N_feat) + "_N_sentences="+ str(N_sentence) + ".p", "wb"))

del predicted_val

print('Predicting probabilites')
predicted_val_proba=log_model.predict_proba(val_data)
pickle.dump(predicted_val_proba, open(parent_dir +"/Stacking step/Predictions_Classifier/Log_proba_val_N_feat="+ str(N_feat) + "_N_sentences="+ str(N_sentence) + ".p", "wb"))

del val_data





#STEP 3: PREDICT ON THE TEST SET
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

predicted_test=log_model.predict(testing_data)
test_error=np.mean(testing_labels != predicted_test)
print("The Logistic Regression testing error is %s" %(test_error))

print('Saving the labels')
pickle.dump(predicted_test, open(parent_dir +"/Stacking step/Predictions_Classifier/Log_predictions_test_N_feat="+ str(N_feat) + "_N_sentences="+ str(N_sentence) + ".p", "wb"))
print('type=')
print(type(predicted_test))

print('Predicting probabilites')
#predicted_train_proba=log_model.predict_proba(training_data)
predicted_test_proba=log_model.predict_proba(testing_data)

#pickle.dump(predicted_train_proba, open(parent_dir +"/Stacking step/Predictions_Classifier/Log_proba_train_N_feat="+ str(N_feat) + "_N_sentences="+ str(N_sentence) + ".p", "wb"))
pickle.dump(predicted_test_proba, open(parent_dir +"/Stacking step/Predictions_Classifier/Log_proba_test_N_feat="+ str(N_feat) + "_N_sentences="+ str(N_sentence) + ".p", "wb"))

