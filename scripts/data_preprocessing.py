import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import string
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.stem.wordnet import WordNetLemmatizer

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
    # lemmatization
    lem = WordNetLemmatizer()
    tokens = [lem.lemmatize(token,"v") for token in tokens]
    sentence = ' '.join(tokens)
    return sentence

# preprocess all comment texts in the trainning set and testing set
train_text = train['comment_text'].copy()
test_text = test['comment_text'].copy()

train_text_clean = [clean_doc(comment) for comment in train_text]
test_text_clean = [clean_doc(comment) for comment in test_text]


# Compare the comment text after preprocessed
print('The 0th comment text in trainning set is:')
print(train.comment_text[0])
print('The 0th comment text in trainning set after preprocessing is:')
print(train_text[0])

# create a tokenizer
max_features = 20000
tokenizer = Tokenizer(num_words = max_features)
tokenizer.fit_on_texts(train_text)
train_encoded = tokenizer.texts_to_sequences(train_text)
val_encoded = tokenizer.texts_to_sequences(val_text)
test_encoded = tokenizer.texts_to_sequences(test_text)
# vocabulary size
print('The vocabulary size is {}.'.format(len(tokenizer.word_index)))
# examples
print(train_encoded[0])

# padding
maxlen = 250
X_train = pad_sequences(train_encoded, maxlen = maxlen, padding = 'post')
X_test = pad_sequences(test_encoded, maxlen = maxlen, padding = 'post')

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
y_train = train[class_names].values
