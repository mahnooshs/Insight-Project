#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 18:54:32 2019

@author: mahnooshsadeghi
"""


import pandas as pd
#import keras

#import codecs

#Read in data
brand = pd.read_json('/Users/mahnooshsadeghi/Desktop/insight/Veritonic/rawdataforveritonicprojects_/brands.json')
industries = pd.read_json('/Users/mahnooshsadeghi/Desktop/insight/Veritonic/rawdataforveritonicprojects_/industries.json')
trans=pd.read_csv('/Users/mahnooshsadeghi/Desktop/insight/Veritonic/rawdataforveritonicprojects_/transcriptions.csv')
tags = pd.read_csv('/Users/mahnooshsadeghi/Desktop/insight/Veritonic/rawdataforveritonicprojects_/tags.csv', error_bad_lines=False)


dataset= pd.merge(tags,trans, on='track_id')


dataset= dataset.rename(columns = {'transcription_text':'Review'})


#Cleaning transcript


dataset = dataset.dropna(axis=0, subset=['Review'])
dataset = dataset.dropna(axis=0, subset=['industry'])


dataset.head()


df= dataset[ ['Review', 'industry']]


df = df[df.industry != '***appears to be a full podcast***']
df = df[df.industry != '**appears to be full podcast**']


df = df.reset_index(drop=True)
df.head()


df.dtypes

import re


corpus = []
for i in range(0, len(df.index)):
    review = re.sub('[^a-zA-Z]', ' ', df['industry'][i])
    review = review.lower()
    corpus.append(review)
    
    

label =[]
for i in range(0, len(df.index)):
    industry=corpus[i].split()[0]
    label.append(industry)
    
    
for i in range (0,len(label)):
    if label[i]== 'after':
        label[i]= 'auto'
    elif label[i]=='building':
        label[i]='home'
    elif label[i]=='legal':
        label[i]='business'
    elif label[i]=='music':
        label[i]='entertainment'
    elif label[i]=='realtors':
        label[i]='real'
    elif label[i]=='restaurant':
        label[i]='qsr'
    elif label[i]=='recruiting': 
        label[i]='job'
    elif label[i]=='supermarkets':
        label[i]='food'
   
        
        
    

myset = set(label)
print(sorted(myset))
len(myset)


label = pd.DataFrame(label)
label.columns=['label']

df['label']=label['label']

df = df[df.label!= 'unknown']
df = df[df.label!= 'good']


df.head()



df.isnull().sum()




####SVC###

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics


svc_model=SVC(gamma='auto')



####vectorizing

df.isnull().sum()

df['label'].value_counts()



X= df['Review']
y=df['label']
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

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions)
accuracy

print(metrics.classification_report(y_test,predictions))




#X_train_tfidf = vectorizer.fit_transform(X_train) # remember to use the original X_train set
#from sklearn.svm import LinearSVC
#clf=LinearSVC()
#clf.fit(X_train_tfidf,y_train)

#from sklearn.linear_model import SGDClassifier

#text_clf = pipeline = Pipeline([
 #   ('vect', CountVectorizer()),
  #  ('tfidf', TfidfTransformer()),
   # ('clf', SGDClassifier(tol=1e-3)),
#])




















