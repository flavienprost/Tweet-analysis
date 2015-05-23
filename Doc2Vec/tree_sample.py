
#Basic Decision Trees
from sklearn import tree
import numpy as np
from scipy import sparse


A = np.array([[1,2,0],[0,0,3],[1,0,4],[1,2,0],[0,0,3],[1,0,4]])
sA = sparse.csr_matrix(A)
sA=sA.todense()


Y = [0,1,1,0,1,0]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(sA, Y)


#To predict class
#print(clf.predict([[2., 2.]]))

#To predict probability
#print("proba equal to")
#print(clf.predict_proba([[2., 2.]]))




#Random Forest

from sklearn.ensemble import RandomForestClassifier

#rfc=RandomForestClassifier(n_estimators=10)
#rfc=rfc.fit(sA, Y)

#Predict class
#print("Predicted class is:")
#print(rfc.predict([0,1,2]))




# #AdaBoost
#from sklearn.cross_validation import cross_val_score
# from sklearn.ensemble import AdaBoostClassifier

# adab_clf = AdaBoostClassifier(n_estimators=100)
# adab_clf=adab_clf.fit(sA,Y)

# print("Predicted class is:")
# print(adab_clf.predict([2.,2.]))

# #scores = cross_val_score(adab_clf, X, Y)
# #scores.mean()     


#SVM

# from sklearn.svm import SVC
# clf = SVC()
# clf.fit(sA, Y)

# print("Predicted class is:")
# print(clf.predict([0,1,2]))
