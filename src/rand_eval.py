import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import speech_recognition as sr

# downloading a set of stop-words
STOPWORDS = set(stopwords.words('english'))

DICT_SIZE = 15000
MAX_LEN = 35

tokenizer = Tokenizer(num_words=DICT_SIZE)

model = tf.keras.models.load_model('model.h5')
model.summary()

emotions_to_labels = {'anger': 0, 'love': 1, 'fear': 2, 'joy': 3, 'sadness': 4,'surprise': 5}
test = pd.read_csv("./test.txt", header=None, sep=";", names=["Lines","Emotions"], encoding="utf-8")

test['Labels'] = test['Emotions'].replace(emotions_to_labels)
y_test = test['Labels'].values

def text_preprocess(text, stop_words=False):
  text = re.sub(r'\W+', ' ', text).lower()
  tokens = word_tokenize(text)
  if stop_words:
    tokens = [token for token in tokens if token not in STOPWORDS]

  return tokens


x_test = [text_preprocess(t, stop_words=True) for t in test['Lines']]
X_test = tokenizer.texts_to_sequences(x_test)
X_test_pad = pad_sequences(X_test, maxlen=MAX_LEN)
results = model.evaluate(X_test_pad, y_test) 
print("test loss, test acc:", results)

def text_preprocess(text, stop_words=False):
  text = re.sub(r'\W+', ' ', text).lower()
  tokens = word_tokenize(text)

  if stop_words:
    tokens = [token for token in tokens if token not in STOPWORDS]
  return tokens

labels_to_emotions = {j:i for i,j in emotions_to_labels.items()}

def predict(texts):
  texts_prepr = [text_preprocess(t) for t in texts]
  sequences = tokenizer.texts_to_sequences(texts_prepr)
  pad = pad_sequences(sequences, maxlen=MAX_LEN)

  predictions = model.predict(pad)
  labels = np.argmax(predictions, axis=1)
  for i, lbl in enumerate(labels):
    print(f'\'{texts[i]}\' --> {labels_to_emotions[lbl]}')



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
    value='null'
  except sr.RequestError as e:
    print("Uh oh! Couldn't request results from Google Speech Recognition service; {0}".format(e))
    value='null'
except KeyboardInterrupt:
  pass
  
test_texts = [value]

predict(test_texts)