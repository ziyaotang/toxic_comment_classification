#import required packages
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from wordcloud import WordCloud ,STOPWORDS
from PIL import Image
from nltk.corpus import stopwords

#settings
color = sns.color_palette()
sns.set_style("whitegrid")
eng_stopwords = set(stopwords.words("english"))

%matplotlib inline

# load text data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print('Trainning set contains {} records and testing set contains {} records.'.format(len(train),len(test)))

train.head()

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

# example comments
all_types = ['clean', 'toxic','severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
for t in all_types:
    record_number = random.randint(1, train[t].sum())
    print('The {}th record of "{}" comments is:'.format(record_number,t))
    print(train[train[t] == 1].iloc[record_number,1])
    print('')

# plot different types of labels
column_sum = pd.Series(labels.sum(axis = 0))
clean = pd.Series(train['clean'].sum(), index = ['clean'])
# add clean document amount as a new type
column_sum = pd.concat([clean, column_sum])
# barplot for different subtypes
plt.figure(figsize=(8,4.5))
ax = sns.barplot(column_sum.index, column_sum.values, palette = "Paired")
plt.xlabel('Subtype', fontsize = 12)
plt.ylabel('number of records', fontsize = 12)
# add the text labels
for rect, label in zip(ax.patches, column_sum.values):
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height(), label,ha = 'center', va='bottom')
# save the figure as .eps file
fig_subtype = ax.get_figure()
fig_subtype.savefig('figure/subtype.eps')
plt.show()

# multi-tagging
multi_tags = [ sum(row_sum == i) for i in range(7)]
multi_tags = pd.Series(multi_tags, index = [0,1,2,3,4,5,6])
# barplot for different labels
plt.figure(figsize=(8,4.5))
ax_1 = sns.barplot(multi_tags.index, multi_tags.values, palette = color_reds)
plt.xlabel('number of tags', fontsize = 12)
plt.ylabel('number of records', fontsize = 12)
# add the text labels
for rect, label in zip(ax_1.patches, multi_tags.values):
        ax_1.text(rect.get_x() + rect.get_width()/2, rect.get_height(), label,ha = 'center', va='bottom')
# save the figure as .eps file
fig_multi_tags = px.get_figure()
fig_multi_tags.savefig('figure/multi_tags.eps')
plt.show()

# crosstab
subtypes = ['severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
i = 1
plt.figure(figsize = (15,4))
for subtype in subtypes:
    plt.subplot(1,5,i)
    cross = sns.heatmap(pd.crosstab(train['toxic'],train[subtype]),annot = True, fmt = '', cbar = False, cmap = "Blues")
    i += 1
plt.show()
# save the figure
fig_cross = cross.get_figure()
fig_cross.savefig('figure/crosstab.eps')

# Comments Length Distribution
# calculate the number of words per comment
n_words = [len(comment) for comment in train.comment_text]

# histogram plot of the number of words each comment contains
fig_n_words = plt.figure()
plt.hist(n_words,bins = np.arange(0,1000,30), color='steelblue')
plt.xlabel('number of words per comment_text')
plt.ylabel('number of comment_texts')
plt.show()
# save the figure
fig_n_words.savefig('figure/n_words.eps')

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
