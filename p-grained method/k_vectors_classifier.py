from sklearn import svm
import cPickle as pickle
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

val_vectors = pickle.load(open("../data/validate_k_vectors.p", "rb"))
val_labels = pickle.load(open("../data/validate_k_labels.p", "rb"))

train_vectors = pickle.load(open("../data/train_k_vectors.p", "rb"))
train_labels = pickle.load(open("../data/train_k_labels.p", "rb"))

test_vectors = pickle.load(open("../data/test_k_vectors.p", "rb"))
test_labels = pickle.load(open("../data/test_k_labels.p", "rb"))

# SVM

"""model_svm = svm.SVC(C=0.1, gamma=0.1)
model_svm.fit(train_vectors, train_labels)
validate_svm = model_svm.predict(val_vectors)
pickle.dump(validate_svm, open("../data/val_predict_svm.p", "wb"))
test_svm = model_svm.predict(test_vectors)
pickle.dump(test_svm, open("../data/test_predict_svm.p", "wb"))

model_knn = KNeighborsClassifier(n_neighbors=100)
model_knn.fit(train_vectors, train_labels)
validate_knn = model_knn.predict(val_vectors)
pickle.dump(validate_knn, open("../data/val_predict_knn.p", "wb"))
test_knn = model_knn.predict(test_vectors)
pickle.dump(test_knn, open("../data/test_predict_knn.p", "wb"))

model_logreg = LogisticRegression(penalty='l2', C=1)
model_logreg.fit(train_vectors, train_labels)
validate_logreg = model_logreg.predict(val_vectors)
pickle.dump(validate_logreg, open("../data/val_predict_logreg.p", "wb"))
test_logreg = model_logreg.predict(test_vectors)
pickle.dump(test_logreg, open("../data/test_predict_logreg.p", "wb"))"""

val = pickle.load(open("../data/val_predict_svm.p", "rb"))
val2 = pickle.load(open("../data/validate_k_labels.p", "rb"))
print((val-val2)[0:500])