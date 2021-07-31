import pandas as pd
import os
import spacy
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from nltk import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import fire
import gensim
import logging
logger = logging.getLogger()
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, recall_score

nlp = spacy.load('en_core_web_sm')

from sklearn.metrics import classification_report, confusion_matrix


TAGS_LIST = ['PERSON', 'WORK_OF_ART', 'ORG', 'LAW', 'MONEY', 'DATE', 'NORP', 'GPE', 'CARDINAL', 'TIME',
             'ORDINAL', 'EVENT', 'PRODUCT', 'FAC', 'LOC', 'PERCENT', 'LANGUAGE', 'QUANTITY']


# CLASSIFIERS_LIST = [MultinomialNB(),
#                     RandomForestClassifier(),
#                     LogisticRegression(),
#                     SGDClassifier(),
#                     KNeighborsClassifier(),
#                     SVC(),
#                     XGBClassifier()]


CLASSIFIERS_LIST = {
        "MultinomialNB": {"class": MultinomialNB, "args": {}, "hyper": {"alpha": [0.1, 0.25, 0.5, 0.75, 1.0]}},
        "AdaBoostClassifier": {"class": AdaBoostClassifier,
                               "args": {"n_estimators": 20, "random_state": 0},
                               "hyper": {"learning_rate": [0.1, 0.25, 0.5, 0.75, 1.0]}},

        "RandomForestClassifier": {"class": RandomForestClassifier,
                                   "args": {"n_estimators": 20, "random_state": 0, "n_jobs": -1},
                                   "hyper": {"criterion": ["gini", "entropy"],
                                             "max_features": ["sqrt", "log2"],
                                             "max_samples": [None, 0.5]
                                             }},

        # "CatBoostClassifier": {"class": CatBoostClassifier, "args": {"n_estimators": 50, "random_state": 0}, "hyper": {
        #     "depth": [6, 8], "l2_leaf_reg": [1.0, 0.2, 3.0, 4.0]
        # }
        #                        },

        'SGDClassifier': {"class": SGDClassifier, "args": {}, "hyper": {
            'alpha': [0.0001, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0],
            'penalty': ['l1', 'l2'],
            'n_jobs': [-1]}},

        'LogisticRegression': {"class": LogisticRegression, "args": {}, "hyper": {
            'C': np.logspace(-4, 4, 20),
            'solver': ['liblinear', ' lbfgs']}},

        'XGBClassifier': {"class": XGBClassifier, "args": {}, "hyper": {
            "eta": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
            "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
            "min_child_weight": [1, 3, 5, 7],
            "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
            "colsample_bytree": [0.3, 0.4, 0.5, 0.7]}},

        # 'SVC': {"class": SVC, "args": {}, "hyper": {
        #     'C': [0.1, 1, 10, 100, 1000],
        #     'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        #     'kernel': ['rbf']}}
    }


class Predictor(object):

    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.pipline = None
        self.clf = None
        self.clf_name = None
        self.enrich_data = None

    def fit(self, X, y):

        self.pipline.fit(X)
        ft_X = self.pipline.transform(X)
        X = EnrichData().fit_transform(X, ft_X)

        self.clf.fit(X, y)

    def predict(self, X):

        ft_X = self.pipline.transform(X)
        X = EnrichData().fit_transform(X, ft_X)

        return self.clf.predict(X)

    def evaluate(self, X_train, y_train, X_test, y_test, clf, enrich_data):

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.pipline = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        self.clf = RandomizedSearchCV(clf['class'](**clf['args']), clf.get("hyper"), cv=3, scoring='accuracy', n_iter=5)
        self.enrich_data = enrich_data
        self.clf_name = clf['class']().__class__.__name__

        self.fit(self.X_train, self.y_train)
        y_pred = self.predict(X_test)

        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        confusion = confusion_matrix(y_test, y_pred)

        return {'clf': self.clf_name,
                'scores': {'f1_score': f1, 'accuracy_score': accuracy, 'recall_score': recall},
                'confusion_matrix': confusion,
                'best_params': self.clf.best_params_}


class EnrichData(object):

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def calc_tags(self, text):
        tags = dict(zip(TAGS_LIST, np.zeros(len(TAGS_LIST))))
        ents = self.nlp.pipe([text], disable=["tagger", "parser"])

        for i in ents:
            for r in i.ents:
                tags[r.label_] += 1

        return tags

    def fit_transform(self, X, ft_X):

        X = pd.DataFrame(X)

        X['tags'] = X.apply(lambda x: self.calc_tags(x['title']), axis=1)
        tags = X['tags'].apply(pd.Series)

        return np.concatenate([ft_X.toarray(), tags.to_numpy()], axis=1)


class Preprocessor(object):

    def __init__(self):
        self.data = None
        self.text_col_name = None
        self.pipline = None

    def lemmatize_stemming(slef, text):
        stemmer = PorterStemmer()
        return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

    def preprocess(self, text):
        result = []
        for token in gensim.utils.simple_preprocess(text):
            result.append(self.lemmatize_stemming(token))

        return ' '.join(result)

    def process_df(self, data, text_col_name):

        self.data = data
        self.text_col_name = text_col_name

        self.data['clean_text'] = self.data.apply(lambda x: self.preprocess(x[self.text_col_name]), axis=1)

        return self.data


def main(import_path, export_path, preprocess=True, enrich_data=False, debug=False):

    file_name = 'clfs_results.xlsx'

    #load data
    X = pd.read_csv(os.path.join(import_path, 'train_X.csv'), usecols=['title'])
    y = pd.read_csv(os.path.join(import_path, 'train_y.csv'), usecols=['Label'])

    if debug:
        X, y = X[0:100], y[0:100]

    #clean text (simple_preprocess + steaming)
    if preprocess:
        file_name = file_name.replace('.xlsx', 'preprocessed.xlsx')

        print('preprocess - START')

        preprocessor = Preprocessor()
        X = preprocessor.process_df(X, 'title')

        print('preprocess - END')

    # split to train / test / validation
    X_train, X_test, y_train, y_test = train_test_split(X['title'], y['Label'], test_size=0.2, random_state=42,
                                                        stratify=y['Label'])

    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42,
    #                                                   stratify=y_train)

    results_list = []

    for clf_name, clf_params in CLASSIFIERS_LIST.items():
        print(f'{clf_name} - START')

        predictor = Predictor()

        clf_evaluation = predictor.evaluate(X_train, y_train, X_test, y_test, clf_params, enrich_data)

        print(f'{clf_name} - END')

        results_list.append(clf_evaluation)

    scores = []

    for clf_result in results_list:

        record = {'clf': clf_result['clf']}
        record.update(clf_result['scores'])
        record.update({'best_params': clf_result['best_params']})
        scores.append(record)

    writer = pd.ExcelWriter(os.path.join(export_path, file_name))

    scores = pd.DataFrame(scores).set_index('clf').sort_values(by='accuracy_score', ascending=False)
    scores.to_excel(writer, sheet_name='daily_activity')

    writer.save()


if __name__ == '__main__':
    fire.Fire(main(import_path='/Users/aloncohen/Yandex/kaggle_compatition',
                   export_path='/Users/aloncohen/Downloads/',
                   preprocess=True, enrich_data=True, debug=False))
