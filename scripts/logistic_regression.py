import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack

# add class names
class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
# load text data
train = pd.read_csv('train.csv').fillna(' ')
test = pd.read_csv('test.csv').fillna(' ')

train_text = train['comment_text']
test_text = test['comment_text']
# create word vectorizer
word_vectorizer = TfidfVectorizer(
    sublinear_tf = True,
    strip_accents = 'unicode',
    analyzer = 'word',
    token_pattern = r'\w{1,}',
    stop_words = 'english',
    ngram_range = (1, 6),
    max_features = 20000)

word_vectorizer.fit(train_text)
train_features = word_vectorizer.transform(train_text)
test_features = word_vectorizer.transform(test_text)

scores = []
submission = pd.DataFrame.from_dict({'id':test['id']})

for class_name in class_names:
    train_target = train[class_name]
    classifier = LogisticRegression(C = 0.2, solver = 'sag')

    cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv = 3, scoring = 'roc_auc'))
    scores.append(cv_score)

    print('CV score for class "{}"" is {}'.format(class_name, cv_score))

    classifier.fit(train_features, train_target)
    # [:,1] for those = 1, which is toxic
    submission[class_name] = classifier.predict_proba(test_features)[:,1]
print('Total CV score is {}'.format(np.mean(scores)))
submission.to_csv('submission.csv', index = False)
