import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import string
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, GRU,Embedding, Dropout, Activation, Flatten, Bidirectional
from keras.layers import SpatialDropout1D, concatenate,Bidirectional, GRU, GlobalAveragePooling1D, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras.layers.convolutional import Conv1D, MaxPooling1D

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score


#settings
stopwords_eng = set(stopwords.words("english"))

# load text data
train = pd.read_csv('../input/train.csv').fillna('')
test = pd.read_csv('../input/test.csv').fillna('')
print('Trainning set contains {} records and testing set contains {} records.'.format(len(train),len(test)))

# this function receives comments and returns clean word-list
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
train_text = train.comment_text.copy()
test_text = test.comment_text.copy()

train_text_clean = [clean_doc(comment) for comment in train_text]
test_text_clean = [clean_doc(comment) for comment in test_text]

max_features = 20000
tokenizer = Tokenizer(num_words = max_features)
tokenizer.fit_on_texts(train_text)
train_encoded = tokenizer.texts_to_sequences(train_text)
test_encoded = tokenizer.texts_to_sequences(test_text)
# vocabulary size
print('The vocabulary size is {}.'.format(len(tokenizer.word_index)))
# examples
print(train_encoded[0])

maxlen = 250
X_train = pad_sequences(train_encoded, maxlen = maxlen, padding = 'post')
X_test = pad_sequences(test_encoded, maxlen = maxlen, padding = 'post')

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
y_train = train[class_names].values

print('X_train shape is {}'.format(X_train.shape))
print('y_train shape is {}'.format(y_train.shape))

# define new callback for ROC AUC score
class roc_callback(Callback):
    def __init__(self,training_data,validation_data):

        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)

        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)

        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))),end=100*' '+'\n')
        return
# function that compile, train and evaluate model
def train_model(model,X_train,y_train,X_val,y_val,batch_size,epochs,filepath):
    # compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    # parameters
    batch_size = batch_size
    epochs = epochs
    # callbacks
    checkpointer =  ModelCheckpoint(filepath=filepath,
                                    verbose=1,
                                    save_best_only=True)
    roc = roc_callback(training_data=(X_train,y_train),validation_data=(X_val,y_val))
    # fit the model
    history = model.fit(X_train,y_train,
                        validation_data=(X_val,y_val),
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[roc,checkpointer])
    # load the model with the best validation loss
    model.load_weights(filepath)
    return model, history

######### change the model here for testing ####################
# build model
model = Sequential()
model.add(Embedding(max_features, 100, input_length=maxlen))
model.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))
model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(600, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(6, activation="sigmoid"))
#summary
model.summary()
###############################################################

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.95, random_state=30)

model_trained, history = train_model(model,
                                     X_train,y_train,
                                     X_val,y_val,
                                     batch_size=512,
                                     epochs=3,
                                     filepath='weights.best.from_scatch_cnn.hdf5')

y_pred = model_trained.predict(X_test,batch_size=1024)
submission = pd.read_csv('../input/sample_submission.csv')
submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
submission.to_csv('submission.csv', index=False)
