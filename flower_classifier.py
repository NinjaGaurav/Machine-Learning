from sklearn.datasets import load_iris
import numpy as np
from sklearn import tree
iris = load_iris()
# Print all features with labels
# for i in range(len(iris.target)):
# 	print('Example %d: Label %s, Features %s'%(i+1, iris.target[i], iris.data[i]))

# testing data
test_idx = [0,1,50,51,100,101]
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis = 0)

# train classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data,train_target) 

print(test_target)
print(clf.predict(test_data))