import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
import string
from collections import Counter
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras import initializers, regularizers, constraints, optimizers, layers


# load text data
train = pd.read_csv('train.csv').fillna('')
test = pd.read_csv('test.csv').fillna('')
print('Trainning set contains {} records and testing set contains {} records.'.format(len(train),len(test)))

# function to preprocess text data
def clean_doc(text_record):
    # split tokens by white space
    tokens = text_record.split()
    # remove punctuation from each string
    table = str.maketrans({key: None for key in string.punctuation})
    tokens = [token.translate(table) for token in tokens]
    # remove tokens that are not alphabetic
    tokens = [token for token in tokens if token.isalpha()]
    # convert letters to lower case
    tokens = [token.lower() for token in tokens]
    # remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # remove short words (one letter)
    tokens = [token for token in tokens if len(token) > 1]
    sentence = ' '.join(tokens)
    return sentence

# preprocess all comment texts in the trainning set and testing set
train_text = train['comment_text'].copy()
test_text = test['comment_text'].copy()
for i in range(len(train_text)):
    train_text[i] = clean_doc(train_text[i])
for j in range(len(test_text)):
    test_text[j] = clean_doc(test_text[j])

# Compare the comment text after preprocessed
print('The 0th comment text in trainning set is:')
print(train.comment_text[0])
print('The 0th comment text in trainning set after preprocessing is:')
print(train_text[0])

# create a tokenizer
max_features = 20000
tokenizer = Tokenizer(num_words = max_features)
# fit the tokenizer
tokenizer.fit_on_texts(train_text)

len(tokenizer.word_index)

encoded_train = tokenizer.texts_to_sequences(train_text)
encoded_train[0]

n_words = [len(comment) for comment in encoded_train]
plt.hist(n_words, bins = np.arange(0,410,10))
plt.show()

# padding
maxlen = 200
X_train = pad_sequences(train_encoded, maxlen = maxlen, padding = 'post')
class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
y_train = y = train[class_names].values
# build model
X_train.shape
model = Sequential()
model.add(Embedding(max_features, 100, input_length=maxlen))
model.add(LSTM(60, return_sequences=True,name='lstm_layer'))
model.add(GlobalMaxPool1D())
model.add(Dropout(0.1))
model.add(Dense(50, activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(6, activation="sigmoid"))
#summary
model.summary()

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
batch_size = 32
epochs = 1
model.fit(X_train,y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
