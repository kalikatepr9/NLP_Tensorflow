#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 12:23:49 2020

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
print('\n training_padded[0]',training_padded[0])
print('\n training_padded.shape',training_padded.shape)

#Convolutional Nueral Network
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    #tf.keras.layers.GlobalAveragePooling1D(),  #or use .flatten
    #tf.keras.layers.Flatten(),
    #tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),  #LSTM
    tf.keras.layers.Conv1D(128,5,activation='relu'),   #128 filters each for 5 words
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
 
#for stacked LSTM
#tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
#tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    
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

epochs=range(len(acc))

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
import matplotlib.pyplot as plt
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


#with the LSTM, the training loss drop nicely, but the validation one increased as I continue training. 
#Again, this shows some over fitting in the LSTM network. While the accuracy of the prediction increased, 
#the confidence in it decreased. So you should be careful to adjust your training parameters 
#when you use different network types

model.save("test.h5")