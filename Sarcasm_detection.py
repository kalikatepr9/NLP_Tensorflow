#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 00:19:17 2020

@author: pranavkalikate
"""
#Sarcasm Headline Dataset -Building classifier

import json
import tensorflow as tf
import numpy as np

#getting the data
data = [json.loads(line) for line in open('Sarcasm_Headlines_Dataset.json', 'r')]

#create empty lists
sentences=[]
labels=[]
urls=[]

#append into list
for item in data:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

#required libraries    
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#hyperparameters
vocab_size=10000  #1000 tweak them to get higher accuracy and decreased val_loss
embedding_dim=16  #32
max_length=32     #16
trunc_type='post'
padding_type='post'
oov_tok='<OOV>'
training_size=20000

#get the training and test/validation set
training_sentences=sentences[0:training_size]
testing_sentences=sentences[training_size:]
training_labels=labels[0:training_size]
testing_labels=labels[training_size:]

#tokenizer
tokenizer=Tokenizer(oov_token=oov_tok,num_words=vocab_size)  
tokenizer.fit_on_texts(training_sentences)
word_index=tokenizer.word_index

print(len(word_index))

#text to sequence for training set
training_sequences=tokenizer.texts_to_sequences(training_sentences)  
#padding
training_padded=pad_sequences(training_sequences,maxlen=max_length,
                              padding=padding_type,truncating=trunc_type)

#text to sequence for testing set
testing_sequences=tokenizer.texts_to_sequences(testing_sentences)  
#padding
testing_padded=pad_sequences(testing_sequences,maxlen=max_length,
                              padding=padding_type,truncating=trunc_type)
print('\n sentences[0]',sentences[0])
print('\n padded[0]',padded[0])
print('\n padded.shape',padded.shape)

#Neural Network
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    #tf.keras.layers.GlobalAveragePooling1D(),  #or use .flatten
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
#Compile
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()

#to solve ValueError: Failed to find data adapter that can handle input:
training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

#Training on epochs
num_epochs= 30
history= model.fit(training_padded,training_labels,epochs=num_epochs,
                  validation_data=(testing_padded,testing_labels),verbose=2)

#Plotting
import matplotlib.pyplot as plt
def plot_graphs(history,string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel('Epochs')
    plt.ylabel(string)
    plt.legend([string,'val_'+string])
    plt.show()

plot_graphs(history,'accuracy')
plot_graphs(history,"loss")