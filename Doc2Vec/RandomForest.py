import pickle
import os.path
import numpy as np

w=4
f=400

#Loading the training and testing data
current_dir= os.path.abspath(os.path.dirname(__file__))
parent_dir=os.path.dirname(current_dir)
project_dir=os.path.dirname(parent_dir)
training_data= pickle.load( open( project_dir + '/data/training_features_f'+ str(f) + '_w' + str(w) +'.p', "rb" ) )
training_labels= pickle.load( open( project_dir + '/data/training_labels_f'+ str(f) + '_w' + str(w) +'.p', "rb" ) )

indices_0=[]
indices_2=[]
indices_4=[]

for i in range(0,len(training_labels)):
	current_label='indices_'+ str(training_labels[i])
	exec("%s.append(%i)" % (current_label,i))

print(len(indices_0))
print(len(indices_2))
print(len(indices_4))


#Function to evaluate the error of a model
def compute_error(x, y, model):
 yfit = model.predict(x)
 return np.mean(y != yfit) 




#Random Forest
from sklearn.ensemble import RandomForestClassifier


rfc = RandomForestClassifier(n_estimators=60, max_features=14, max_depth=15)
rfc = rfc.fit(training_data, training_labels)

#Compute the training error
train_error = compute_error(training_data,training_labels,rfc)
print("The random forest training error is %s" %(train_error))

#Compute the testing error
testing_data= pickle.load( open( project_dir + '/data/testing_features_f'+ str(f) + '_w' + str(w) +'.p', "rb" ) )
testing_labels= pickle.load( open( project_dir + '/data/testing_labels_f'+ str(f) + '_w' + str(w) +'.p', "rb" ) )

indices_test_0=[]
indices_test_2=[]
indices_test_4=[]

for i in range(0,len(testing_labels)):
	current_label='indices_test_'+ str(testing_labels[i])
	exec("%s.append(%i)" % (current_label,i))

indices_test_0_4=indices_test_0+indices_test_4

print(len([testing_data[x] for x in indices_test_0_4]))
print(len(indices_test_0))

test_error = compute_error([testing_data[x] for x in indices_test_0_4], [testing_labels[x] for x in indices_test_0_4],rfc)
print("The random forest testing error is %s" %(test_error))

#To predict probability
#print("proba equal to")
#print(clf.predict_proba([[2., 2.]]))

