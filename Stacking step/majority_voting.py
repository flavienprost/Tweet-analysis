import pickle
import os.path
import numpy as np
import time
from sklearn import linear_model


N_feat=10000
N_sentence=50000
Classifier=['Log']
#Classifier2=['knn','bin_naivebayes','svm','logreg']
Classifier2=['bin_naivebayes','svm']
#Classifier2=['bin_naivebayes','svm']

current_dir= os.path.abspath(os.path.dirname(__file__))
parent_dir=os.path.dirname(current_dir)

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
classif_predictions=pickle.load(open(current_dir+ '/Predictions_Classifier/'+Classifier[0]+ "_predictions_test_N_feat="+ str(N_feat) + "_N_sentences="+ str(N_sentence)+ ".p", "rb"))

for i in range(0,len(classif_predictions)):
    features_test_stacking.append([classif_predictions[i]])

for clf in Classifier:
    if clf<>Classifier[0]:
        classif_predictions=pickle.load(open(current_dir+ '/Predictions_Classifier/'+clf+ "_predictions_test_N_feat="+ str(N_feat) + "_N_sentences="+ str(N_sentence) + ".p", "rb"))

        for k in range(0,len(classif_predictions)):
            features_test_stacking[k].append(classif_predictions[k])


for clf2 in Classifier2:
        classif_predictions=pickle.load(open(current_dir+ '/Predictions_Classifier/test_predict_'+clf2+'.p', "rb"))
        print(classif_predictions)
        for k in range(0,len(classif_predictions)):
            features_test_stacking[k].append(classif_predictions[k])




test_predictions=np.zeros(len(testing_labels))

for i in range(0,len(testing_labels)):
    sum=0

    for j in range(0,len(Classifier)+len(Classifier2)):
        sum=sum+features_test_stacking[i][j]
    if sum>float(len(Classifier)+len(Classifier2)/2):
        test_predictions[i]=1


test_error=np.mean(testing_labels!= test_predictions)

print("The majority voting testing error (stacking step) is %s" %str(test_error))