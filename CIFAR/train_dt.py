from sklearn import tree
import numpy as np
from load_data import unpickle
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
import sys

train_path = 'cifar-100-python/train'
test_path = 'cifar-100-python/test'
train = unpickle(train_path)
test = unpickle(test_path)

print(train['data'].shape)

train_data = np.array(train['data'])
train_label = np.array(train['fine_labels'])
test_data = np.array(test['data'])
test_label = np.array(test['fine_labels'])

num_class = int(sys.argv[1])
train_sub_data = []
train_sub_label = []
test_sub_data = []
test_sub_label = []

for i in range(len(train_label)):
    if train_label[i] < num_class:
        train_sub_label.append(train_label[i])
        train_sub_data.append(train_data[i])

if __name__ == '__main__':

    clf = tree.DecisionTreeClassifier()

    n_split = 5
    kfold = KFold(n_splits=n_split)
    train_sub_sub_data = np.array(train_sub_data)
    train_sub_sub_label = np.array(train_sub_label)
    count = 0
    for train_index, test_index in kfold.split(train_sub_sub_data):
        clf.fit(train_sub_sub_data[train_index], train_sub_sub_label[train_index])
        print(accuracy_score(clf.predict(train_sub_sub_data[test_index]), train_sub_sub_label[test_index]))

    scores = cross_val_score(clf, train_sub_sub_data, train_sub_sub_label, cv=kfold, n_jobs=-1)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
