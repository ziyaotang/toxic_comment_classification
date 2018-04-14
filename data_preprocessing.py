import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
import string
import enchant

# load text data
train = pd.read_csv('train.csv').fillna('')
test = pd.read_csv('test.csv').fillna('')
print('Trainning set contains {} records and testing set contains {} records.'.format(len(train),len(test)))

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
    # remove non-english words
    # American English words dictionary
    en_US = enchant.Dict("en_US")
    # British English words dictionary
    en_GB = enchant.Dict("en_GB")
    tokens = [token for token in tokens if (en_US.check(token) or en_GB.check(token))]
    # remove short words (one letter)
    tokens = [token for token in tokens if len(token) > 1]
    sentence = ' '.join(tokens)
    return sentence

# save the cleaned text into a file
def save_preprocessed(preprocessed_text, file_name):
    file_name = open(file_name,'wb')
    lines = '\n'.join(preprocessed_text).encode('utf-8').strip()
    file_name.write(lines)
    file_name.close()

# preprocess all comment texts in the trainning set
train_text_clean = []
train_text = train['comment_text']
for comment in train_text:
    sentence = clean_doc(comment)
    train_text_clean.append(sentence)
len(train_text_clean)
# compare the comment text before and after preprocessing
train_text_clean[0]
train['comment_text'][0]
# save the cleaned text into a file
save_preprocessed(train_text_clean, 'train_text.txt')

# preprocess all comment texts in the testing set
test_text_clean = []
test_text = test['comment_text']
for comment in test_text:
    sentence = clean_doc(comment)
    test_text_clean.append(sentence)
len(test_text_clean)
# compare the comment text before and after preprocessing
test_text_clean[0]
test['comment_text'][0]
# save the cleaned text into a file
save_preprocessed(test_text_clean, 'test_text.txt')
