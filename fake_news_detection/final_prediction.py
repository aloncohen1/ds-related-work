import fire
import os
import pandas as pd
import gensim
import gensim.corpora as corpora
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from nltk.corpus import stopwords
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk import WordNetLemmatizer
from sklearn.preprocessing import StandardScaler

nlp = spacy.load('en_core_web_sm')
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

TAGS_LIST = ['PERSON', 'WORK_OF_ART', 'ORG', 'LAW', 'MONEY', 'DATE', 'NORP', 'GPE', 'CARDINAL', 'TIME',
             'ORDINAL', 'EVENT', 'PRODUCT', 'FAC', 'LOC', 'PERCENT', 'LANGUAGE', 'QUANTITY']


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
        for token in gensim.utils.simple_preprocess(self.remove_stop_words(text), deacc=True):
            result.append(self.lemmatize_stemming(token))

        return ' '.join(result)

    def remove_stop_words(self, text):
        result = [i for i in text.split() if i not in stop_words]

        return ' '.join(result)

    def process_df(self, data, text_col_name):

        self.data = data
        self.text_col_name = text_col_name

        self.data['clean_text'] = self.data.apply(lambda x: self.preprocess(x[self.text_col_name]), axis=1)

        return self.data


class Tagger(object):

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def calc_tags(self, text):
        tags = dict(zip(TAGS_LIST, np.zeros(len(TAGS_LIST))))
        ents = self.nlp.pipe([text], disable=["tagger", "parser"])

        for i in ents:
            for r in i.ents:
                tags[r.label_] += 1

        return tags

    def enrich_with_tags(self, X, vectorize_X):

        X = pd.DataFrame(X)

        X['tags'] = X.apply(lambda x: self.calc_tags(x['title']), axis=1)
        tags = X['tags'].apply(pd.Series)

        return np.concatenate([vectorize_X.toarray(), tags.to_numpy()], axis=1)


def main(import_path, export_path):

    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

    X_train = pd.read_csv(os.path.join(import_path, 'train_X.csv'), usecols=['title'])
    y_trian = pd.read_csv(os.path.join(import_path, 'train_y.csv'), usecols=['Label'])
    X_test = pd.read_csv(os.path.join(import_path, 'test_X.csv'), usecols=['title'])
    index = pd.read_csv(os.path.join(import_path, 'test_X.csv'), usecols=['index'])['index']

    X_train['status'] = 'train'
    X_test['status'] = 'test'

    all_data = pd.concat([X_train, X_test]).reset_index(drop=True)

    print('preprocess - START')

    preprocessor = Preprocessor()
    all_data = preprocessor.process_df(all_data, 'title')

    data_for_topics = all_data.clean_text.values.tolist()
    data_for_topics = [i.split() for i in data_for_topics]

    id2word = corpora.Dictionary(data_for_topics)

    corpus = [id2word.doc2bow(text) for text in data_for_topics]

    print('preprocess - END')

    print('LDA - START')

    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=11,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)

    print('LDA - END')

    lda_output = [lda_model.get_document_topics(i, minimum_probability=1e-5) for i in corpus]
    topics_features = []

    for item in lda_output:
        items = []
        for prob in item:
            items.append(prob[1])
        topics_features.append(items)

    topics_features = np.array(topics_features)

    train_X = all_data[all_data['status'] == 'train']
    test_X = all_data[all_data['status'] == 'test']

    topics_train = topics_features[train_X.index]
    topics_test = topics_features[test_X.index]

    tf_idf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))

    X_train_transform = tf_idf.fit_transform(train_X['clean_text'])
    X_test_transform = tf_idf.transform(test_X['clean_text'])

    print('Tagger - START')

    train_X = Tagger().enrich_with_tags(train_X, X_train_transform)
    test_X = Tagger().enrich_with_tags(test_X, X_test_transform)

    print('Tagger - END')

    train_X = np.hstack([train_X, topics_train])
    test_X = np.hstack([test_X, topics_test])

    print('fit - START')

    clf = SGDClassifier(penalty='l2', n_jobs=-1, alpha=0.0001)

    clf.fit(train_X, y_trian['Label'].values)

    print('fit - END')

    prediction = clf.predict(test_X)

    prediction = pd.DataFrame(prediction, index=index, columns=['Label'])

    prediction.to_csv(os.path.join(export_path, 'prediction.csv'))

    print('Finish All!')


if __name__ == '__main__':
    fire.Fire(main(import_path='/Users/aloncohen/Yandex/kaggle_compatition',
                   export_path='/Users/aloncohen/Downloads/'))