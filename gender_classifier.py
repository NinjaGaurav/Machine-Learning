from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

X = [	#Height,Weight,Shoe size
		[180,80,44],
		[177,70,43],
		[160,60,38],
		[154,54,37],
		[166,65,40],
		[190,90,47],
		[175,65,39],
		[177,70,40],
		[159,66,41],
		[177,80,45],
		[181,85,43],
]
# Y[i] tells whether record holder X[i] is Male or Female 
Y = ['male','male','female','female','male','male','male',
'female','male','male','male']
# Test data in test below:
test = [[150,61,36],[180,81,46],[165,80,39]]

#########################################################
clf = DecisionTreeClassifier()
clf = clf.fit(X,Y)

prediction = clf.predict(test)

print('Decision Tree: ',prediction)
#########################################################
clf = RandomForestClassifier(n_estimators=2)
clf = clf.fit(X,Y)

prediction = clf.predict(test)

print('Random Forest Classifier: ',prediction)
#########################################################
clf = GaussianNB()
clf = clf.fit(X,Y)

prediction = clf.predict(test)

print('Naive Bayes:', prediction)
##########################################################
clf = LogisticRegression()
clf = clf.fit(X,Y)

prediction = clf.predict(test)

print('Logistic Regression:', prediction)

#########################################################
clf = KNeighborsClassifier(n_neighbors=3)
clf = clf.fit(X,Y)

prediction = clf.predict(test)

print('K Nearest Neighbor:', prediction)