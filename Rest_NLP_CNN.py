#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 20:00:02 2020

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



#Steps:
#1- Clean text
#2- create a bag of words model
#3 - apply ML model(Classification) onto Bag of words

# Importing the dataset
df = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

#value counts
df.Review.value_counts()

#Getting the numeric features with null values
nulls_train = pd.DataFrame(df.isnull().sum().sort_values(ascending=False)[:80])

#length of the each string in Review column in each entry
df['pre_clean_len'] = [len(t) for t in df.Review]

#hyperparameters
vocab_size=5000  # tweak them to get higher accuracy and decreased val_loss
embedding_dim=32  #32 dimension words that are found together are given similar vectors.
#ex. dull & boring / fun & exciting
max_length=150    #of text
trunc_type='post'
padding_type='post'
oov_tok='<OOV>'
training_size=800
#embedding means the vectors for each word with their associated sentiment.


#get the training and test/validation set
training_sentences=df.Review[0:training_size]
testing_sentences=df.Review[training_size:]
training_labels=df.Liked[0:training_size]
testing_labels=df.Liked[training_size:]

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

print(testing_padded[0])
print(testing_padded.shape)

#Covolutional Neural Network
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    #tf.keras.layers.GlobalAveragePooling1D(),  #or use .flatten
    #tf.keras.layers.Flatten(),
    #tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),  #LSTM
    tf.keras.layers.Conv1D(128,5,activation='relu'),   #128 filters each for 5 words
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),  #24 neurons
    tf.keras.layers.Dense(1, activation='sigmoid')
])
#word_embedding means words and associated words are clustered as vectors in multi-dimensional space.
#Embedding means to establish meaning from them.

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
"""
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)

#Reverse word index for getting proper key:value pair
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(training_padded[3]))
print(training_sentences[3])


#Helper function to plot in projector
import io

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
      word = reverse_word_index[word_num]
      embeddings = weights[word_num]
      out_m.write(word + "\n")
      out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()       
"""

#to solve ValueError: Failed to find data adapter that can handle input:
training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)


#Training on epochs
num_epochs= 50
history= model.fit(training_padded,training_labels,epochs=num_epochs,
                  validation_data=(testing_padded,testing_labels),verbose=2)

model.save("test.h5")

#Plotting
import matplotlib.image  as mpimg

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


#with the LSTM, the training loss drop nicely, but the validation one increased as I continue training. 
#Again, this shows some over fitting in the LSTM network. While the accuracy of the prediction increased, 
#the confidence in it decreased. So you should be careful to adjust your training parameters 
#when you use different network types