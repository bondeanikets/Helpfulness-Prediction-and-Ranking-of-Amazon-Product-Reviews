
"""
Created on Tue Nov  1 22:21:52 2016

@author: Aniket
"""

import pandas as pd
import numpy as np
from os import path
import matplotlib.pyplot as plt
from operator import add
from wordcloud import WordCloud
import pickle
import time

start = time.time()

global fig
fig = 0

f = open('store_truncateddata.pckl', 'rb')
df_reviews = pickle.load(f)
f.close()

"""1. This plots the histogram of Overall Ratings of The Products"""
plt.figure(fig)
fig = fig + 1
f = plt.hist(np.array(df_reviews.overall), bins = 16, color = 'r', alpha = .8)
plt.margins(0.1, 0.1)
plt.title('Distribution of the Overall Ratings of The Products')
plt.ylabel('Number of Reviews with the Ratings')
plt.xlabel('Ratings')
plt.savefig('1.Histogram_Overall_Rating_truncateddata.png', dpi=900, bbox_inches='tight')

"""2. This plots the Distribution of Helpfulness Reviews"""
plt.figure(fig)
fig = fig + 1        
plt.hist(np.nan_to_num(np.array(df_reviews.percent_helpful)), bins = 10, color = 'r', alpha = .8)
plt.margins(0.1, 0.1)
plt.title('Distribution of Helpfulness of The Products')
#plt.ylabel('Number of Reviews with the Helpfulness Ratings')
plt.yscale('linear')
plt.ylabel('Number of Reviews')
plt.xlabel('Helpfulness in Percent')
plt.savefig('2.Distribution_of_Helpfulness_truncateddata.png', dpi=900, bbox_inches='tight')

"""3. This plots the histogram of Helpfulness Reviews"""
plt.figure(fig)
fig = fig + 1
plt.hist(np.array(df_reviews.review_helpful), bins = 16, color = 'r', alpha = .8)
plt.margins(0.1, 0.1)
plt.xticks([0, 1])
plt.title('Helpfulness of The Products')
plt.ylabel('Number of Reviews with the Helpfulness Ratings')
plt.xlabel('Helpfulness')
plt.savefig('3.Histogram_Helpfulness_truncateddata.png', dpi=900, bbox_inches='tight')

"""4. This plots the Overall Ratings VS the Helpfulness of the Reviews"""
a = np.zeros((1,5))
b = np.zeros((1,5))
d = np.zeros((1,5))
c = np.nan_to_num(np.array(df_reviews.percent_helpful))
index = np.array(list (range(5))) + 1  
bar_width = 0.35
for i in range(len(df_reviews)):
    for j in range(5):
        if (df_reviews.overall[i] == j+1):
            if (c[i] > 0 and c[i] <= 25):
                a[0][j] = a[0][j] + 1
            elif (c[i] >= 75 and c[i] <= 100):
                b[0][j] = b[0][j] + 1
            else:
                d[0][j] = d[0][j] + 1
                
a1 = np.divide((a*100), np.add(a,b,d))
b1 = np.divide((b*100), np.add(a,b,d))
plt.figure(fig)
fig = fig + 1
plt.bar(index, a1[0], bar_width, color = 'g', alpha = .8, label = 'Not Helpful (<40%)')
plt.bar(index + bar_width, b1[0], bar_width, color = 'r', alpha = .8, label = 'Helpful (>60%)')
plt.ylim(0,100)
plt.xticks(index + bar_width, ('1', '2', '3', '4', '5'))
plt.margins(0.1, 0.1)
plt.legend(loc='right', prop={'size':10})
plt.title(r'Percent of Reviews Found Helpful/Unhelpful' + "\n" + r'Among Voted Reviews by Rating', multialignment='center')
plt.ylabel('Percent')
plt.xlabel('Overall Ratings (Out of 5)')
plt.savefig('4.Overall Ratings VS the Helpfulness_truncateddata.png', dpi=900, bbox_inches='tight')
plt.show()

"""5, 6. This Plots the Popular Words in the Review With Respect
to Positive and Negative Reviews using Wordcloud Package"""

negative = list(filter(lambda i:df_reviews.overall[i] < 3, range(len(df_reviews))))
text = open("Negative.txt", "w")
text.write(''.join(df_reviews.reviewText[negative]))
text.close()

positive = list(filter(lambda i:df_reviews.overall[i] > 3, range(len(df_reviews))))
text = open("Positive.txt", "w")
text.write(''.join(df_reviews.reviewText[positive]))
text.close()

wordcloud_negative = WordCloud(background_color="white").generate(open(path.join(r'E:\Users\Dell\Desktop\Projects Material\Project Workspace', 'Negative.txt')).read())
plt.figure(fig)
fig = fig + 1
plt.imshow(wordcloud_negative)
plt.axis("off")
plt.savefig('5.Wordcloud for Negative Reviews_truncateddata.png', dpi=1200)

wordcloud_positive = WordCloud(background_color="white").generate(open(path.join(r'E:\Users\Dell\Desktop\Projects Material\Project Workspace', 'Positive.txt')).read())
plt.figure(fig)
fig = fig + 1
plt.imshow(wordcloud_positive)
plt.axis("off")
plt.savefig('6.Wordcloud for Positive Reviews_truncateddata.png', dpi=1200)

"""7. Plots Histogram of Number of words"""
plt.figure(fig)
fig = fig + 1
plt.hist(df_reviews.no_of_words, bins=200, color = 'm')
plt.margins(0.1, 0.1)
#plt.xlim([0, max(df_reviews.no_of_words)])
#plt.ylim([0, max(df_reviews.no_of_words) + 10])
plt.xlabel('Number of words in reviews')
plt.ylabel('Number of reviews, in the range of Words per review')
plt.title('Histogram of Number of words')
plt.savefig('7.Histogram of Number of words_truncateddata.png', dpi=900, bbox_inches='tight')

"""8. This Plots the Variance of Polarity with respect to Overall Ratings"""
plt.figure(fig)
fig = fig + 1
plot = df_reviews.boxplot('polarity', 'overall', patch_artist=True)
plt.savefig('8.Variance of Polarity with respect to Overall Ratings_truncateddata.png', dpi=900, bbox_inches='tight')

"""9. Plots Number of Sentences VS Helpfulness"""
plt.figure(fig)
fig = fig + 1
df_reviews.plot(kind='scatter', x='percent_helpful', y='no_of_sentences', color = 'm')
plt.xlim([0, 100])
plt.ylim([0, max(df_reviews.no_of_sentences) + 10])
plt.xlabel('Helpfulness in percent')
plt.ylabel('Number of Sentences per review')
plt.title('Number of Sentences VS Helpfulness')
plt.savefig('9.Number of Sentences VS Helpfulness_truncateddata.png', dpi=900, bbox_inches='tight')


print('It took {0:0.2f} seconds'.format(time.time() - start))