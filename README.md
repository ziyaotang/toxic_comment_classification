## Toxic Comment Classification

This project is about online comments classification, which is based on a public Kaggle competition [Toxic
Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge). The goal of this project is to build a multi-headed model that’s capable of detecting different types of toxicity like threats, obscenity, insults, and identity-based hate. Such model will hopefully help online discussion become more productive and respectful.

Several model architectures have been studied, such as CNN, RNN and pre-trained embedding (GloVe). Please check the `capstone_report.pdf` for detailed discussion.
![WordCloud for clean online comment](figure/wordcloud_clean.eps)
### Dataset
The dataset contains about 160k human labeled comments from Wikipedia Talk pages. The labeled annotations
are obtained by asking 5000 crowd-workers to rate Wikipedia comments according to their toxicity (likely to
make others leave the conversation).The dataset can be obtained from:
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data


### Install
This project requires **Python 3** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [Keras](https://keras.io/)
- [WordCloud](https://github.com/amueller/word_cloud)
- [NLTK](https://www.nltk.org/)

You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html)

### Code

The code is provided in the `toxic_comment_classification.ipynb` notebook file. The scripts are provided in the `scripts` folder as for data exploration, data preprocessing and model training.
