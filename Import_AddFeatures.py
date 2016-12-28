
"""
Created on Tue Nov  1 22:11:01 2016

@author: Aniket
"""

from textblob import TextBlob
import pandas as pd
import numpy as np
import pickle

import time

start = time.time()


def parse(path):
    g = open(path, 'r')
    for l in g:
        yield eval(l)
      
time_dict={}
count_dict={}

"""This function extracts information out of the tuple 'helpful' 
column so that we can start to create some other features"""
def creating_basic_features():
    global df_reviews
    df_reviews['helpful_votes'] = df_reviews.helpful.apply(lambda x: x[0])
    df_reviews['overall_votes'] = df_reviews.helpful.apply(lambda x: x[1])
    #gives non-consecutive indexes to dataframe as overall_votes < 5 are eliminated    
    df_reviews = df_reviews[df_reviews.overall_votes >= 5]    
    #make index of dataframe consecutive    
    df_reviews['i1'] = pd.Series(range(len(df_reviews)), index=df_reviews.index)
    df_reviews = df_reviews.set_index('i1')
    df_reviews['percent_helpful'] = round((df_reviews['helpful_votes'] / df_reviews['overall_votes']) * 100)
    df_reviews['review_helpful'] = np.where((df_reviews.percent_helpful > 60), 1, 0)
    
    #Computations for the unixreviewtime of a review with respect to the first review on the same product
    global time_dict
    global count_dict
    for i in range(len(df_reviews)):
        pid = df_reviews['asin'][i]
        tim=df_reviews['unixReviewTime'][i]
        if pid in time_dict:
            count_dict[pid]=count_dict[pid]+1
            if tim<time_dict[pid]:
                time_dict[pid]=tim
        else:
            count_dict[pid]=1
            time_dict[pid]=tim
    
    #Subtract each products' first revviews' unixReviewTime from correspoding reviews
    a = np.zeros((1,10619))
    for i in range(len(df_reviews)):
        pid = df_reviews['asin'][i]
        a[0][i] = df_reviews.unixReviewTime[i] - time_dict[pid]     
    df_reviews.unixReviewTime = a.transpose()

def create_textblob_features():
    global df_reviews
    df_reviews['polarity'] = df_reviews['reviewText'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df_reviews['no_of_words'] = df_reviews['reviewText'].apply(lambda x: len(TextBlob(x).words))
    df_reviews['no_of_sentences'] = df_reviews['reviewText'].apply(lambda x: len(TextBlob(x).sentences))
    df_reviews['subjectivity'] = df_reviews['reviewText'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    df_reviews['words_per_sentence'] = df_reviews.no_of_words / df_reviews.no_of_sentences
    #df_reviews['sentence_complexity'] = df_reviews.reviewText.apply(lambda x: float(len(set(TextBlob(x).words))) / float(len(TextBlob(x).words)))

def delete_unnecessary_features():
   '''
   ID of the product is not needed as the information is already used
   '''
   del df_reviews['asin']
    
   '''
   helpful is no longer required as its information is extracted in 
   'helpful_votes' and 'overall_votes'
   '''
   del df_reviews['helpful']

   '''
   'reviewTime' is nt needed as it contains redundant information
   which is same as 'unixReviewTime'
   '''
   del df_reviews['reviewTime']

   '''
   Removing the features that are not required
   '''
   del df_reviews['reviewerID']
   del df_reviews['reviewerName']

    

reviews_data = r'E:\Users\Dell\Desktop\Projects Material\Dataset\Cell_Phones_and_Accessories_5.json'
reviews = ([])

obj_reviews = parse(reviews_data)
for i in obj_reviews:
    reviews.append(i)

df_reviews = pd.DataFrame(reviews)
creating_basic_features()
create_textblob_features()
delete_unnecessary_features()

f = open('store_truncateddata.pckl', 'wb')
pickle.dump(df_reviews, f)
f.close()


print('It took {0:0.2f} seconds'.format(time.time() - start))