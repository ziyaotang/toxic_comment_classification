import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords

# settings
stopwords_eng = set(stopwords.words("english"))

# load text data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print('Trainning set contains {} records and testing set contains {} records.'.format(len(train),len(test)))

#
train.head()

sns.palplot(sns.light_palette("green"))
