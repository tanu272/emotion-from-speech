#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import re 
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import speech_recognition as sr

lb = LabelEncoder()
model = tf.keras.models.load_model("model.h5")
stopwords = set(nltk.corpus.stopwords.words('english'))
vocab_size = 10000
len_sentence = 150


def text_prepare_text(text):
    stemmer = PorterStemmer()
    corpus = []
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    text = " ".join(text)
    print(text)

    one_hot_word = one_hot(input_text=text, n=vocab_size)
    print([one_hot_word])
    embeddec_doc = pad_sequences(sequences=[one_hot_word],
                              maxlen=len_sentence,
                              padding="pre")
    # # print(text.shape)
    return embeddec_doc


## recognize audio
r = sr.Recognizer()
m = sr.Microphone()

try:
    with m as source: 
        r.adjust_for_ambient_noise(source)
    #print("Set minimum energy threshold to {}".format(r.energy_threshold))
    print("Say something!")
    with m as source: 
        audio = r.listen(source)
    print("Got it! Now to recognize it...")
    try:
        value = r.recognize_google(audio, language='en-GB')
        print(value)
    except sr.UnknownValueError:
        print("Oops! Didn't catch that")
    except sr.RequestError as e:
        print("Uh oh! Couldn't request results from Google Speech Recognition service; {0}".format(e))
except KeyboardInterrupt:
    pass

##use audio statement to check speaker's feelings
mod_text = text_prepare_text(value)
mod_text.shape


a=(model.predict(mod_text))
b=(model.predict(mod_text).argmax())
#print(a)
print(b)


y=['anger','fear','joy','love','sadness','surprise']
lb.fit(y)

final_pred=lb.inverse_transform(b.ravel())
print(final_pred)

