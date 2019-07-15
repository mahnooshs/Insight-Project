#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 07:45:39 2019

@author: mahnooshsadeghi
"""

import pandas as pd
import numpy as np
#import keras
import nltk
import re
#import codecs

#Read in data
brand = pd.read_json('/Users/mahnooshsadeghi/Desktop/insight/Veritonic/rawdataforveritonicprojects_/brands.json')
industries = pd.read_json('/Users/mahnooshsadeghi/Desktop/insight/Veritonic/rawdataforveritonicprojects_/industries.json')
trans=pd.read_csv('/Users/mahnooshsadeghi/Desktop/insight/Veritonic/rawdataforveritonicprojects_/transcriptions.csv')
tags = pd.read_csv('/Users/mahnooshsadeghi/Desktop/insight/Veritonic/rawdataforveritonicprojects_/tags.csv', error_bad_lines=False)


dataset= pd.merge(tags,trans, on='track_id')


dataset= dataset.rename(columns = {'transcription_text':'Review'})


#Cleaning transcript

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
dataset = dataset.dropna(axis=0, subset=['Review'])
dataset = dataset.reset_index(drop=True)

dataset.head()


df= dataset[ ['Review', 'industry', 'has_call_to_action','ad_type', 'has_disclaimer', 'words_per_second',
            'has_music', 'track_duration','word_count', 'number_of_voices',
            'voice_over_gender']]

df.head()


df['has_disclaimer'].value_counts()

df = df[df.has_disclaimer != 'yes,no']
df = df.dropna(axis=0, subset=['has_disclaimer'])
df = df[df.words_per_second != 138414]

df = df.dropna(axis=0, subset=['word_count'])

df['word_count'] = pd.to_numeric(df['word_count'], errors='raise', downcast=None)

df = df.reset_index(drop=True)

df.isnull().sum()


import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize


sentencecount =[]
for i in range (0, len(df.index)):
    sentences = df.iloc[i,0]
    n = sent_tokenize(sentences)
    sentencecount.append(len(n))


df ['sentence']= pd.DataFrame (sentencecount)



checkcount =[]
for i in range (0, len(df.index)):
    sentences= df.iloc[i,0]
    if ' dot' in sentences:
        count= 1
    else:
        count=0
    checkcount.append(count)    

df ['checkcount']= pd.DataFrame (checkcount)







count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))


punct=[]
for i in range (0,len(df.index)):
    sentences=df.iloc[i,0]
    n = count(sentences, string.punctuation)
    punct.append(n)

df ['punct']= pd.DataFrame (punct)


import matplotlib.pyplot as plt
%matplotlib inline

#plt.xscale('log')
bins = 1.15**(np.arange(0,50))
plt.hist(df[df['has_disclaimer']=='no']['word_count'],bins=bins,alpha=0.8)
plt.hist(df[df['has_disclaimer']=='yes']['word_count'],bins=bins,alpha=0.8)
plt.legend(('no','yes'))
plt.show()


bins = 1.15**(np.arange(0,15))
plt.hist(df[df['has_disclaimer']=='no']['words_per_second'],bins=bins,alpha=0.8)
plt.hist(df[df['has_disclaimer']=='yes']['words_per_second'],bins=bins,alpha=0.8)
plt.legend(('no','yes'))
plt.show()


bins = 1.15**(np.arange(0,30))
plt.hist(df[df['has_disclaimer']=='no']['track_duration'],bins=bins,alpha=0.8)
plt.hist(df[df['has_disclaimer']=='yes']['track_duration'],bins=bins,alpha=0.8)
plt.legend(('no','yes'))
plt.show()



plt.hist(df[df['has_disclaimer']=='no']['checkcount'])
plt.hist(df[df['has_disclaimer']=='yes']['checkcount'])
plt.legend(('no','yes'))
plt.show()



bins = 1.15**(np.arange(0,30))
plt.hist(df[df['has_disclaimer']=='no']['punct'],bins=bins,alpha=0.8)
plt.hist(df[df['has_disclaimer']=='yes']['punct'],bins=bins,alpha=0.8)
plt.legend(('no','yes'))
plt.show()




bins = 1.15**(np.arange(0,30))
plt.hist(df[df['has_disclaimer']=='no']['sentence'],bins=bins,alpha=0.8)
plt.hist(df[df['has_disclaimer']=='yes']['sentence'],bins=bins,alpha=0.8)
plt.legend(('no','yes'))
plt.show()



#Violin shaped diagrams
import seaborn as sns
sns.set(style="ticks", color_codes=True)

sns.catplot(x="words_per_second", y="has_disclaimer", 
            kind="violin", data=df);


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# X is feature data
X= df[['words_per_second', 'word_count', 'punct','checkcount','sentence']]

y= df[['has_disclaimer']]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

from sklearn.linear_model import LogisticRegression
lr_model= LogisticRegression(solver='lbfgs')
lr_model.fit(X_train,y_train)

from sklearn import metrics
predictions=lr_model.predict(X_test)

print(metrics.confusion_matrix(y_test,predictions))

Confusion = pd.DataFrame(metrics.confusion_matrix(y_test,predictions), index=['disclaimer','no disclaimer'], columns=['disclaimer','no disclaimer'])
Confusion

accuracy = accuracy_score(y_test, predictions)

##### Under sampling over sampling ###NEMUDAR Monaseb


#import xgboost
#from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

target_count = df.has_disclaimer.value_counts()
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / (target_count[1]+target_count[0]), 2))

target_count.plot(kind='bar', title='Count (Disclaimer)');





############Udersampling
# Class count 
count_class_0, count_class_1 = df.has_disclaimer.value_counts()

# Divide by class
df_class_0 = df[df['has_disclaimer'] =='no']
df_class_1 = df[df['has_disclaimer'] == 'yes']

df_class_0_under = df_class_0.sample(count_class_1)
df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)

print('Random under-sampling:')
print(df_test_under.has_disclaimer.value_counts())

df_test_under.has_disclaimer.value_counts().plot(kind='bar', 
                                 title='Count (target)');



                                         
                                         
#Logistic regression

X= df_test_under[['words_per_second', 'word_count', 'punct','checkcount','sentence']]

y= df_test_under[['has_disclaimer']]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

from sklearn.linear_model import LogisticRegression
lr_model= LogisticRegression(solver='lbfgs')
lr_model.fit(X_train,y_train)

from sklearn import metrics
predictions=lr_model.predict(X_test)

print(metrics.confusion_matrix(y_test,predictions))

Confusion = pd.DataFrame(metrics.confusion_matrix(y_test,predictions), 
                         index=['disclaimer','no disclaimer'], 
                         columns=['disclaimer','no disclaimer'])
Confusion

accuracy = accuracy_score(y_test, predictions)
accuracy

print(metrics.classification_report(y_test,predictions))

####SVC###

from sklearn.svm import SVC

svc_model=SVC(gamma='auto')
svc_model.fit(X_train,y_train)

predictions=svc_model.predict(X_test)

print(metrics.confusion_matrix(y_test,predictions))

Confusion = pd.DataFrame(metrics.confusion_matrix(y_test,predictions),
                         index=['disclaimer','no disclaimer'], 
                         columns=['disclaimer','no disclaimer'])
Confusion

accuracy = accuracy_score(y_test, predictions)
accuracy


####vectorizing

df.isnull().sum()

df['has_disclaimer'].value_counts()

from sklearn.model_selection import train_test_split


X= df['Review']
y=df['has_disclaimer']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
from sklearn.feature_extraction.text import CountVectorizer
count_vect=CountVectorizer()
#Fit vectorizer to the data (buid the vocab, count the number of words)
#count_vect.fit(X_train)
#X_train_counts = count_vect.transform(X_train)

# Transform the original text message --> vector
X_train_counts = count_vect.fit_transform(X_train)
 
X_train.shape

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()
X_train_tfidf=tfidf_transformer.fit_transform(X_train_counts)


#other way

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train) # remember to use the original X_train set
from sklearn.svm import LinearSVC
clf=LinearSVC()
clf.fit(X_train_tfidf,y_train)


#With pipeline
from sklearn.pipeline import Pipeline

text_clf = Pipeline ([('tfidf', TfidfVectorizer()), ('clf', LinearSVC())])

text_clf.fit (X_train, y_train)


predictions = text_clf.predict(X_test)

Confusion = pd.DataFrame(metrics.confusion_matrix(y_test,predictions), index=['disclaimer','no disclaimer'], columns=['disclaimer','no disclaimer'])
Confusion

accuracy = accuracy_score(y_test, predictions)
accuracy

print(metrics.classification_report(y_test,predictions))











