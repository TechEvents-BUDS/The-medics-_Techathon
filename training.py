import os
import json
import numpy as np
import nltk
import pickle
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
import random

lemmatizer = WordNetLemmatizer()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

intents_path = os.path.join(BASE_DIR, 'intents.json')
words_path = os.path.join(BASE_DIR, 'words.pkl')
classes_path = os.path.join(BASE_DIR, 'classes.pkl')
model_path = os.path.join(BASE_DIR, 'chatbot_model.h5')

intents = json.loads(open(intents_path).read())

words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = sorted(list(set([lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words])))
classes = sorted(list(set(classes)))

pickle.dump(words, open(words_path, 'wb'))
pickle.dump(classes, open(classes_path, 'wb'))

training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]
    for w in words:
        bag.append(1 if w in pattern_words else 0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(train_y[0]), activation='softmax')
])

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
model.save(model_path)
print("Training complete. Model saved as chatbot_model.h5")
