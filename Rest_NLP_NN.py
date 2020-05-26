#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 20:57:15 2020

@author: pranavkalikate
"""

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
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers

#Steps:
#1- Clean text
#2- Build a model
#3 - apply ML model(Classification) onto Bag of words

# Importing the dataset
df = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

#value counts
df.Review.value_counts()

#Getting the numeric features with null values
nulls_train = pd.DataFrame(df.isnull().sum().sort_values(ascending=False)[:80])

#length of the each string in Review column in text column in each entry
df['pre_clean_len'] = [len(t) for t in df.Review]

#hyperparameters # tweak them to get higher accuracy and decreased val_loss
vocab_size=5000  #vocubulary of known words
embedding_dim=32  #32
max_length=150    #of review to keep
trunc_type='post' 
padding_type='post' 
oov_tok='<OOV>'  #Out of word vocabulary
training_size=800  #80%

#get the training and test/validation set
training_sentences=df.Review[0:training_size]
testing_sentences=df.Review[training_size:]

training_labels=df.Liked[0:training_size]
testing_labels=df.Liked[training_size:]

#tokenizer                          #assigns tokens for each unique word
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer=Tokenizer(oov_token=oov_tok,num_words=vocab_size)  
tokenizer.fit_on_texts(training_sentences)
word_index=tokenizer.word_index   #key-value pair

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

#Neural Network
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),  #or use .flatten
    #tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
#word_embedding means words and associated words are clustered as vectors in multi-dimensional space.
#Embedding means to establish meaning from them.

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