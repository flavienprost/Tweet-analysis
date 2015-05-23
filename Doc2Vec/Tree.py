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

#print(len(training_data[1]))
#print((training_data[1]))
#print(training_data[1][1])
#print(training_labels[1])
#print(len(training_data))



#Function to evaluate the error of a model
def compute_error(x, y, model):
 yfit = model.predict(x)
 return np.mean(y != yfit) 




#Basic Decision Trees
from sklearn import tree


clf = tree.DecisionTreeClassifier()
clf = clf.fit(training_data, training_labels)

#Compute the training error
train_error = compute_error(training_data,training_labels,clf)
print("The tree training error is %s" %(train_error))

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


test_error = compute_error([testing_data[x] for x in indices_test_0_4], [testing_labels[x] for x in indices_test_0_4],clf)
print("The tree testing error is %s" %(test_error))

#To predict probability
#print("proba equal to")
#print(clf.predict_proba([[2., 2.]]))

