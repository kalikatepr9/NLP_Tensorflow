#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 10:53:44 2019

@author: pranavkalikate
"""

# Natural Language Processing

#Steps:
#1- Clean text
#2- create a bag of words model
#3 - apply ML model(Classification) onto Bag of words

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts to get rid of unneccessary words     #compulsory step in NLP
"""  
#This is for first review..run code line by line to see the results..

import re                               #library has sequence of characters that define a search patterns  
import nltk                             #library which will help to remove irrelevent words
nltk.download('stopwords')              #all the stopwords present in review will be removed
from nltk.corpus import stopwords       #load the stopwords package
from nltk.stem.porter import PorterStemmer   #required for stemmimg
review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][0]) # ^ dont remove, removed letter will be replaced by space ' '
review = review.lower()                 #covert to lower case        
review = review.split()                 #words into lists
ps = PorterStemmer()
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]  #set is recommended for larger files such as documents
review = ' '.join(review)
"""

import re      
import nltk   
nltk.download('stopwords')   
from nltk.corpus import stopwords  
from nltk.stem.porter import PorterStemmer  
corpus = []                                  #corpus is a collection of cleaned text
for i in range(0, 1000):                     #dataset have thousand reviews
        review = re.sub('[^a-zA-Z]',' ', dataset['Review'][i]) 
        review = review.lower()    
        review = review.split()  
        ps = PorterStemmer()    
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] #List Comprehension
        review = ' '.join(review) 
        corpus.append(review)    
        
from sklearn.feature_extraction.text import CountVectorizer 
"""
# Creating the Bag of Words model (feature extraction)
from sklearn.feature_extraction.text import CountVectorizer  
cv = CountVectorizer(max_features = 1000)

#Variables  
X_1 = cv.fit_transform(corpus).toarray()   #Sparse matrix
y = dataset.iloc[:, 1].values  """

#OR

#n-gram
ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2)) #2 words
#ngram_vectorizer.fit(corpus)
X = ngram_vectorizer.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values 

# Splitting the dataset into the Training set and Test set       
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

"""
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
"""

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Accuracy = 72%
#Precision = 85%
#Recall = 54%
#F1 Score = 66%

#TF-IDF    #to evaluate the importance of a word to a document in a collection or corpus
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
tfidf_vectorizer = TfidfVectorizer()
values = tfidf_vectorizer.fit_transform(corpus)

# Show the Model as a pandas DataFrame
feature_names = tfidf_vectorizer.get_feature_names()
tfidf_score=pd.DataFrame(values.toarray(), columns = feature_names)

#Predicting the new results
new_review= "Food was not tasty"
review = re.sub('[^a-zA-Z]', " ",new_review)
review = review.lower()
review = review.split()
ps = PorterStemmer()
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
new_review = [new_review]
new_review = ngram_vectorizer.transform(new_review).toarray()
new_y_pred = classifier.predict(new_review)

#another review
new_review= "Quality taste"
review = re.sub('[^a-zA-Z]', " ",new_review)
review = review.lower()
review = review.split()
ps = PorterStemmer()
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
new_review = [new_review]
new_review = ngram_vectorizer.transform(new_review).toarray()
new_y_pred_2 = classifier.predict(new_review)

#another review
new_review= "I dont like the taste"
review = re.sub('[^a-zA-Z]', " ",new_review)
review = review.lower()
review = review.split()
ps = PorterStemmer()
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
new_review = [new_review]
new_review = ngram_vectorizer.transform(new_review).toarray()
new_y_pred_3 = classifier.predict(new_review)

#another review
new_review= "I hate the flavour"
review = re.sub('[^a-zA-Z]', " ",new_review)
review = review.lower()
review = review.split()
ps = PorterStemmer()
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
new_review = [new_review]
new_review = ngram_vectorizer.transform(new_review).toarray()
new_y_pred_4 = classifier.predict(new_review)