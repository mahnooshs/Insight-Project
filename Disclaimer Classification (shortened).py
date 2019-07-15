#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 07:45:39 2019

@author: mahnooshsadeghi
"""

import pandas as pd

#import nltk


#Read in data
brand = pd.read_json('/Users/mahnooshsadeghi/Desktop/insight/Veritonic/rawdataforveritonicprojects_/brands.json')
industries = pd.read_json('/Users/mahnooshsadeghi/Desktop/insight/Veritonic/rawdataforveritonicprojects_/industries.json')
trans=pd.read_csv('/Users/mahnooshsadeghi/Desktop/insight/Veritonic/rawdataforveritonicprojects_/transcriptions.csv')
tags = pd.read_csv('/Users/mahnooshsadeghi/Desktop/insight/Veritonic/rawdataforveritonicprojects_/tags.csv', error_bad_lines=False)


dataset= pd.merge(tags,trans, on='track_id')


dataset= dataset.rename(columns = {'transcription_text':'Review'})


#Cleaning transcript


#nltk.download('stopwords')

dataset = dataset.dropna(axis=0, subset=['Review'])
dataset = dataset.reset_index(drop=True)

dataset.head()


df= dataset[ ['Review', 'has_disclaimer']]

df.head()


df['has_disclaimer'].value_counts()

df = df[df.has_disclaimer != 'yes,no']
df = df.dropna(axis=0, subset=['has_disclaimer'])

df = df.reset_index(drop=True)

df.isnull().sum()


####SVC###

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics


svc_model=SVC(gamma='auto')



####vectorizing

df.isnull().sum()

df['has_disclaimer'].value_counts()



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
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions)
accuracy

print(metrics.classification_report(y_test,predictions))











