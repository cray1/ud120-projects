#!/opt/conda/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
""" 
    
import sys
import collections
from sklearn.svm import SVC
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

#########################################################
clf= SVC(C=10000.0,kernel='rbf')
#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 
t0 = time()
clf.fit(features_train,labels_train)
print("train in: "+ str(round(time()-t0,3)) + "s")

t1= time()
prediction = clf.predict(features_test)  
counter=collections.Counter(prediction)
print(counter)

count =0
for item in prediction:
    if item == 1:
        count+=1
print(count)

print("accuracy of: " + str(accuracy_score(labels_test, prediction, normalize=True)) + " in: " + str(round(time()-t1,3)) + "s")


