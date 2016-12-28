
"""
Created on Sun Nov 27 05:14:11 2016

@author: Aniket
"""

import pickle
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm 
import pandas as pd

f = open('store_truncateddata_for_ranking_1.pckl', 'rb')
df_reviews = pickle.load(f)
f.close()

'''make index of dataframe consecutive'''    
df_reviews['i1'] = pd.Series(range(len(df_reviews)), index=df_reviews.index)
df_reviews = df_reviews.set_index('i1')
    
'''keep only necessary features'''
df_reviews = df_reviews[['reviewText', 'percent_helpful', 'helpful_votes', 'overall_votes', 'review_helpful']]

df_reviews['percent_helpful'] = (df_reviews['percent_helpful'] / 100)

   
