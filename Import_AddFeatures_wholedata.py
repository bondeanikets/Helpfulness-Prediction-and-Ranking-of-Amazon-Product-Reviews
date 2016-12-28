
"""
Created on Wed Oct 26 20:28:29 2016

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
        
"""This function extracts information out of the tuple 'helpful' 
column so that we can start to create some other features"""
def creating_basic_features():    
    df_reviews['helpful_votes'] = df_reviews.helpful.apply(lambda x: x[0])
    df_reviews['overall_votes'] = df_reviews.helpful.apply(lambda x: x[1])
    df_reviews['percent_helpful'] = round((df_reviews['helpful_votes'] / df_reviews['overall_votes']) * 100)
    df_reviews['review_helpful'] = np.where((df_reviews.percent_helpful > 60) & (df_reviews.overall_votes > 5), 1, 0)

def create_textblob_features():
    df_reviews['polarity'] = df_reviews['reviewText'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df_reviews['no_of_words'] = df_reviews['reviewText'].apply(lambda x: len(TextBlob(x).words))
    df_reviews['no_of_sentences'] = df_reviews['reviewText'].apply(lambda x: len(TextBlob(x).sentences))
    df_reviews['subjectivity'] = df_reviews['reviewText'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    df_reviews['words_per_sentence'] = df_reviews.no_of_words / df_reviews.no_of_sentences
    #df_reviews['sentence_complexity'] = df_reviews.reviewText.apply(lambda x: float(len(set(TextBlob(x).words))) / float(len(TextBlob(x).words)))
 
reviews_data = r'E:\Users\Dell\Desktop\Projects Material\Dataset\Cell_Phones_and_Accessories_5.json'
reviews = ([])

obj_reviews = parse(reviews_data)
for i in obj_reviews:
    reviews.append(i)

df_reviews = pd.DataFrame(reviews)
creating_basic_features()
create_textblob_features()

f = open('store_wholedata.pckl', 'wb')
pickle.dump(df_reviews, f)
f.close()


print('It took {0:0.2f} seconds'.format(time.time() - start))