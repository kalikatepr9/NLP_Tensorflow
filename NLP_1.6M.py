#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:45:10 2020

@author: pranavkalikate
"""

#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import tensorflow as tf
import csv
import random

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers

#Import Dataset
cols = ['sentiment','id','date','query_string','user','text']
#df = pd.read_csv('training.1600000.processed.noemoticon.csv',encoding='latin-1',header=None, names=cols)
df=pd.read_csv('training_cleaned.csv',header=None,names=cols)
#cleaned the Stanford dataset to remove LATIN1 encoding to make it easier for Python CSV reader

#value counts
df.sentiment.value_counts()

#Drop unnecesary features
df.drop(['id','date','query_string','user'],axis=1,inplace=True)
#neg_class=df[df.sentiment == 0]
#pos_class=df[df.sentiment == 4]

#length of the string in text column in each entry
df['pre_clean_len'] = [len(t) for t in df.text]

#Hyperparameters
embedding_dim = 100
max_length = 50     #text_size
trunc_type='post'   #default pre
padding_type='post'
oov_tok = "<OOV>"   #for out of vocabulary
training_size=160000
test_portion=.1

num_sentences = 0
corpus=[]  #corpus contains list of review and label

with open('training_cleaned.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        list_item=[]
        list_item.append(row[5])
        list_label=row[0]
        if list_label=='0':
            list_item.append(0)
        else:
            list_item.append(1)
        num_sentences = num_sentences + 1
        corpus.append(list_item)

print(num_sentences)
print(len(corpus))
print(corpus[1])

#Tokenizing and padding
sentences=[]
labels=[]
random.shuffle(corpus)
for x in range(training_size):
    sentences.append(corpus[x][0])
    labels.append(corpus[x][1])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

#getting the word index 
word_index = tokenizer.word_index
vocab_size=len(word_index)

#sequencing
sequences = tokenizer.texts_to_sequences(sentences)

#padding
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

split = int(test_portion * training_size)  #test size

#splitting into training and testing
test_sequences = padded[0:split]
training_sequences = padded[split:training_size]
test_labels = labels[0:split]
training_labels = labels[split:training_size]

#to check the word index
print(vocab_size)
print(word_index['nice'])

#Neural Network
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),  #or use .flatten
    #tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
    
#Compile
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
#try other NNs as well & can try combination of NN.

"""
#neural network
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length, weights=[embeddings_matrix], trainable=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=4),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
"""
model.summary()

#to solve ValueError: Failed to find data adapter that can handle input:
training_sequences = np.array(training_sequences)
training_labels = np.array(training_labels)
test_sequences = np.array(test_sequences)
test_labels = np.array(test_labels)

num_epochs = 5
history = model.fit(training_sequences, training_labels, epochs=num_epochs, validation_data=(test_sequences, test_labels), verbose=2)
print("Training Complete")

#Plotting
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r')
plt.plot(epochs, val_acc, 'b')
plt.title('Training and validation accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Accuracy", "Validation Accuracy"])
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r')
plt.plot(epochs, val_loss, 'b')
plt.title('Training and validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Loss", "Validation Loss"])

plt.figure()


# Expected Output
# A chart where the validation loss does not increase sharply!