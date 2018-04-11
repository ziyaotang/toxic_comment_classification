%pwd
%cd toxic_comment_classification/

#import required packages
#basics
import pandas as pd
import numpy as np

#misc
import gc
import time
import warnings

#stats
from scipy.misc import imread
from scipy import sparse
import scipy.stats as ss

#viz
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from wordcloud import WordCloud ,STOPWORDS
from PIL import Image
import matplotlib_venn as venn

#nlp
import string
import re    #for regex
import nltk
from nltk.corpus import stopwords
import spacy
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
# Tweet tokenizer does not split at apostophes which is what we want
from nltk.tokenize import TweetTokenizer


#FeatureEngineering
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split





#settings
start_time=time.time()
color = sns.color_palette()
sns.set_style("whitegrid")
eng_stopwords = set(stopwords.words("english"))
warnings.filterwarnings("ignore")

lem = WordNetLemmatizer()
tokenizer=TweetTokenizer()

%matplotlib inline


# load text data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print('Trainning set contains {} records and testing set contains {} records.'.format(len(train),len(test)))

train.tail(10)

# class imbalance
labels = train.iloc[:,2:]
row_sum = labels.sum(axis = 1)
# mark comments without any labels as "clean"
train['clean'] = (row_sum == 0)
print('Total comments is {}.'.format(len(train)))
print('The clean comments is {}, which is {} of total comments.'.format(train['clean'].sum(), train['clean'].sum()/len(train)))
# check for missing values in the trainning set
missing_train = train.isnull().sum()
missing_test = test.isnull().sum()

# new function: barplot with text labels for each column
def bar_plot_label(x, y, x_axis, y_axis, filename):
    plt.figure(figsize=(8,4.5))
    ax = sns.barplot(x, y)
    plt.xlabel(x_axis, fontsize = 12)
    plt.ylabel(y_axis, fontsize = 12)
    # add the text labels
    for rect, label in zip(ax.patches, y):
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height(), label,ha = 'center', va='bottom')
    # save the figure as .eps file
    fig = ax.get_figure()
    fig.savefig(filename)
    plt.show()

# plot different types of labels
column_sum = labels.sum(axis = 0)
# add clean document amount as a new type
column_sum['clean'] = train['clean'].sum()
# barplot for different labels
plot_subtype = bar_plot_label(column_sum.index, column_sum.values, 'Subtype', 'number of records','subtype.eps')

# multi-tagging
multi_tags = [ sum(row_sum == i) for i in range(7)]
multi_tags = pd.Series(multi_tags, index = [0,1,2,3,4,5,6])
bar_plot_label(multi_tags.index, multi_tags.values)

# correlation between subtypes
corr = train.iloc[:,2:8].corr()
plt.figure(figsize = (12,8))
corr_plot = sns.heatmap(corr, annot = True)
# save the figure
fig = corr_plot.get_figure()
fig.savefig('correlation.eps')
