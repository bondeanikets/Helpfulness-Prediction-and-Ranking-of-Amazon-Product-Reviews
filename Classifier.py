
"""
Created on Wed Nov  2 11:23:22 2016

@author: Aniket
"""

import pickle
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn import svm 

f = open('store_truncateddata.pckl', 'rb')
df_reviews = pickle.load(f)
f.close()


del df_reviews['reviewText']
del df_reviews['summary']
del df_reviews['helpful_votes']
del df_reviews['overall_votes']
del df_reviews['percent_helpful']

train,test = train_test_split(df_reviews, train_size=0.8)

train_targets = train['review_helpful'].values
test_targets = test['review_helpful'].values

del train['review_helpful']
del test['review_helpful']

train = np.nan_to_num(np.array(train))
test = np.nan_to_num(np.array(test))


model_LR = LogisticRegression()
model_LR.fit(train, train_targets)
predictions_LR = model_LR.predict(test)
print ('Logistic Regression')
print ('Accuracy score: '+str(accuracy_score(test_targets,predictions_LR)))
print ('Precision score: '+str(precision_score(test_targets,predictions_LR)))

modelTree = DecisionTreeClassifier(max_depth=7)
modelTree.fit(train, train_targets)
predictions_Tree = modelTree.predict(test)
print ('Decision Tree')
print ('Accuracy score: '+str(accuracy_score(test_targets,predictions_Tree)))
print ('Precision score: '+str(precision_score(test_targets,predictions_Tree)))


modelGr = GradientBoostingClassifier()
modelGr.fit(train, train_targets)
predictions_Gr = np.array(modelGr.predict(test))
print ('Gradient Boosting')
print ('Accuracy score: '+str(accuracy_score(test_targets,predictions_Gr)))
print ('Precision score: '+str(precision_score(test_targets,predictions_Gr)))

#
#model_SVM = svm.SVC(kernel='rbf')
#model_SVM.fit(train, train_targets)
#predictions_SVM = model_SVM.predict(test)
#print ('SVM')
#print ('Accuracy score: '+str(accuracy_score(test_targets,predictions_SVM)))
#print ('Precision score: '+str(precision_score(test_targets,predictions_SVM)))