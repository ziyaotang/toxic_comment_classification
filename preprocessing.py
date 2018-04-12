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
import random

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
from sklearn.metrics import confusion_matrix





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
column_sum = pd.Series(labels.sum(axis = 0))
clean = pd.Series(train['clean'].sum(), index = ['clean'])
# add clean document amount as a new type
column_sum = pd.concat([clean, column_sum])
# barplot for different labels
plot_subtype = bar_plot_label(column_sum.index, column_sum.values, 'Subtype', 'number of records','figure/subtype.eps')

# multi-tagging
multi_tags = [ sum(row_sum == i) for i in range(7)]
multi_tags = pd.Series(multi_tags, index = [0,1,2,3,4,5,6])
bar_plot_label(multi_tags.index, multi_tags.values,'number of tags', 'number of records', 'figure/multi_tags.eps')

# correlation between subtypes
corr = train.iloc[:,2:8].corr()
plt.figure(figsize = (12,8))
corr_plot = sns.heatmap(corr, annot = True)
# save the figure
fig_corr = corr_plot.get_figure()
fig_corr.savefig('figure/correlation.eps')

# crosstab
subtypes = ['severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
i = 1
plt.figure(figsize = (15,4))
for subtype in subtypes:
    plt.subplot(1,5,i)
    cross = sns.heatmap(pd.crosstab(train['toxic'],train[subtype]),annot = True, fmt = '', cbar = False, cmap = "Blues")
    i += 1
plt.show()
fig_cross = cross.get_figure()
fig_cross.savefig('figure/crosstab.eps')

# example comments
all_types = ['clean', 'toxic','severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
for t in all_types:
    record_number = random.randint(1, train[t].sum())
    print('The {}th record of "{}" comments is:'.format(record_number,t))
    print(train[train[t] == 1].iloc[record_number,1])
    print('')

# frequent words
def wordcloud_generator(subtype, background, mask_file):
    subset = train[train[subtype] == True]
    subset_text = subset.comment_text.values
    # import mask file
    mask = np.array(Image.open(mask_file))
    mask = mask[:, :, 1]
    # create a wordcloud for the text
    wordcloud = WordCloud(background_color = background, mask = mask, stopwords = stopwords_eng)
    wordcloud.generate(" ".join(subset_text))
    return wordcloud

# wordcloud plot for clean comments
wordcloud_clean = wordcloud_generator('clean','white','image/mask_clean.jpg')
wordcloud_clean_plot = plt.imshow(wordcloud_clean.recolor(colormap = 'viridis', random_state = 29), interpolation = 'bilinear')
plt.axis("off")
plt.show()
#save the figure
fig_wordcloud_clean = wordcloud_clean_plot.get_figure()
fig_wordcloud_clean.savefig('figure/wordcloud_toxic.eps')

# wordcloud plot for subtypes of toxic comments
toxic_types = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
colormaps = ['magma', 'Reds', 'inferno', 'Paired_r', 'Paired_r', 'plasma']
plt.subplots(3,2, figsize=(10,15))
i = 1
for toxic_type, colormap in zip(toxic_types, colormaps):
    plt.subplot(3,2,i)
    wordcloud_toxic = wordcloud_generator(toxic_type,'black','image/mask_toxic.jpg')
    wordcloud_toxic_plot = plt.imshow(wordcloud_toxic.recolor(colormap = colormap, random_state = 29), interpolation = 'bilinear')
    plt.axis("off")
    plt.gca().set_title(toxic_type)
    i += 1
plt.show()
# save the figure
fig_wordcloud_toxic = wordcloud_toxic_plot.get_figure()
fig_wordcloud_toxic.savefig('figure/wordcloud_toxic.eps')
