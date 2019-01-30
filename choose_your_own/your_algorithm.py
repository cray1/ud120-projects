#!/opt/conda/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData 
from class_vis import prettyPicture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
# grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
# bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
# grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
# bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


# #### initial visualization
# plt.xlim(0.0, 1.0)
# plt.ylim(0.0, 1.0)
# plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
# plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
# plt.legend()
# plt.xlabel("bumpiness")
# plt.ylabel("grade")
# plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary
##kneighbors
clf = KNeighborsClassifier(n_neighbors=10, algorithm='ball_tree', weights='distance')


#Adaboost
# from sklearn.svm import SVC
# svc=SVC(C=100.0,kernel='rbf')

# clf= AdaBoostClassifier(n_estimators=10, base_estimator=svc,learning_rate=1, algorithm='SAMME')
# 


# Random forest
#clf = RandomForestClassifier(n_estimators=50, criterion='gini', )


clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
accuracy= accuracy_score(labels_test,pred)
print(accuracy)




#try:
prettyPicture(clf, features_test, labels_test)
#except NameError:
#    pass
