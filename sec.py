#!/usr/bin/env python
# coding: utf-8


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import matplotlib.pyplot as plt
import seaborn as sns
import re 
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder


test_data = pd.read_csv("./test.txt", header=None, sep=";", names=["Comment","Emotion"], encoding="utf-8")
train_data = pd.read_csv("./train.txt", header=None, sep=";", names=["Comment","Emotion"], encoding="utf-8")
val_data = pd.read_csv("./val.txt", header=None, sep=";", names=["Comment","Emotion"], encoding="utf-8")




print("Train : ", train_data.shape)
print("Test : ", test_data.shape)
print("Validation : ", val_data.shape)


"""sns.set()
sns.countplot(train_data["Comment"])
plt.show()"""



train_data["length"] = [len(i) for i in train_data["Comment"]]
plt.plot(train_data["length"], color = "green")




sns.kdeplot(x=train_data["length"], hue=train_data["Emotion"])



from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
train_data["Emotion"] = lb.fit_transform(train_data["Emotion"])
test_data["Emotion"] = lb.fit_transform(test_data["Emotion"])
val_data["Emotion"] = lb.fit_transform(val_data["Emotion"])




train_data.head()


test_data.head()


val_data.head()


vocab_size = 10000
train_data["length"].max()


train_data["length"].min()
len_sentence = 150
train_data.head()


stopwords = set(nltk.corpus.stopwords.words('english'))




def text_prepare(data, column):
    print(data.shape)
    stemmer = PorterStemmer()
    corpus = []
    
    for text in data[column]:
        text = re.sub("[^a-zA-Z]", " ", text)
        
        text = text.lower()
        text = text.split()
        
        text = [stemmer.stem(word) for word in text if word not in stopwords]
        text = " ".join(text)
        
        corpus.append(text)
    one_hot_word = [one_hot(input_text=word, n=vocab_size) for word in corpus]
    embeddec_doc = pad_sequences(sequences=one_hot_word,
                              maxlen=len_sentence,
                              padding="pre")
    print(data.shape)
    return embeddec_doc
        


x_train=text_prepare(train_data, "Comment")
x_validate=text_prepare(val_data, "Comment")
x_test=text_prepare(test_data, "Comment")



x_train.shape



y_train=train_data["Emotion"]
y_validate=val_data["Emotion"]
y_test=test_data["Emotion"]



enc = OneHotEncoder()
y_train = np.array(y_train)
y_train = enc.fit_transform(y_train.reshape(-1,1)).toarray()
y_test = np.array(y_test)
y_validate = np.array(y_validate)
y_test = enc.fit_transform(y_test.reshape(-1,1)).toarray()
y_validate = enc.fit_transform(y_validate.reshape(-1,1)).toarray()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam


model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=150, input_length=len_sentence))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(64, activation="sigmoid"))
model.add(Dropout(0.2))
model.add(Dense(6, activation="softmax"))



model.compile(optimizer="Adam", loss = "categorical_crossentropy", metrics=["accuracy"])


es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 5)
mc = ModelCheckpoint('./model.h5', monitor = 'val_accuracy', mode = 'max', verbose = 1, save_best_only = True)
y_train.shape


hist = model.fit(x_train, y_train, epochs = 25, batch_size = 64, validation_data=(x_validate, y_validate),verbose = 1, callbacks= [es, mc])
