#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 19:43:00 2019

@author: mahnooshsadeghi
"""

#Import pandas
import pandas as pd

#Read in data
brand = pd.read_json('/Users/mahnooshsadeghi/Desktop/insight/Veritonic/rawdataforveritonicprojects_/brands.json')
industries = pd.read_json('/Users/mahnooshsadeghi/Desktop/insight/Veritonic/rawdataforveritonicprojects_/industries.json')
trans=pd.read_csv('/Users/mahnooshsadeghi/Desktop/insight/Veritonic/rawdataforveritonicprojects_/transcriptions.csv')
tags = pd.read_csv('/Users/mahnooshsadeghi/Desktop/insight/Veritonic/rawdataforveritonicprojects_/tags.csv', error_bad_lines=False)

#adjusting column names to start merging
#list(brand)
#brand= brand.rename(columns = {'classification_id':'track_id'})

dataset= pd.merge(tags,trans, on='track_id')


dataset= dataset.rename(columns = {'transcription_text':'Review'})


#Cleaning transcript

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
dataset = dataset.dropna(axis=0, subset=['Review'])
dataset = dataset.reset_index(drop=True)


import string
corpus=[]
for i in range(0,len(dataset.index)):
    review= dataset['Review'][i].translate(str.maketrans('', '', string.punctuation))
    review=review.lower()
    corpus.append(review)



corpusb=[]
for i in range(0,len(brand.index)):
    brands= brand['name'][i].translate(str.maketrans('', '', string.punctuation))
    brands=brands.lower()
    corpusb.append(brands)


        
corpusl=[]

for i in range (0,len(corpus)):
    label=''
    for j in range (0,len(corpusb)):
        if (' '+corpusb[j]+' ') in corpus[i]:
            label= label+' '+corpusb[j]
    corpusl.append(label)


data= pd.DataFrame (corpusl)
import numpy as np
np.array(data[0]=='').sum()


#df = pd.DataFrame(({'A': [0,1,2], 'B': ['indeed','blackrock']}))



    
corpusupper=[]
for i in range(0,len(dataset.index)):
    review= dataset['Review'][i].translate(str.maketrans('', '', string.punctuation))
    corpusupper.append(review)

import spacy
nlp = spacy.load('en_core_web_sm')
taggi=[]
for i in range (0, len(corpusupper)):
    org=''
    x = nlp (corpusupper[i])
    if x.ents:
        for ent in x.ents:
            if ent.label_=='ORG':
                org= ent.text+ ','+ org
    taggi.append(org)
    
    
data= pd.DataFrame (taggi)
np.array(data[0]=='').sum()



join = pd.concat([pd.DataFrame(corpus), pd.DataFrame(corpusl), pd.DataFrame(taggi),], axis=1)


join.columns = ['trans', 'tag1', 'tagml']

for i in range (0, len (join.index)):
    if (join.tagml[i]==''):
        sentences = join.iloc[i,0]
        if ' dot' in sentences:
            before_keyword, keyword, after_keyword = sentences.partition('dot')
            first, *middle, last = before_keyword.split()
            join.tagml[i] =last


join.tagml = np.where(join.tagml=='',  join.tag1, join.tagml)


###################

from spacy import displacy
displacy.render (nlp(corpusupper[1]), style= 'ent')

colors= {'ORG': 'radial-gradient(yellow,green)'}
options = {'ents':['ORG'], 'colors':colors}

for i in range (0,5):
    displacy.render (nlp(corpusupper[i]), style= 'ent', jupyter=True, options=options)
    
    

displacy.serve(nlp(corpusupper[0]),style='ent', options=options)