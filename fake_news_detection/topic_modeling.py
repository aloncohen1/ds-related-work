import fire
# Run in python console
import nltk
from sklearn.preprocessing import StandardScaler
import re
import pandas as pd
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from nltk.corpus import stopwords
import numpy as np
import tqdm
# spacy for lemmatization
import spacy
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)





def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence).encode('utf-8'), deacc=True))  # deacc=True removes punctuations


# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def compute_coherence_values(corpus, dictionary, k, a):
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=dictionary,
                                                num_topics=k,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha=a,
                                                per_word_topics=True)

    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')

    return coherence_model_lda.get_coherence()

def clean_text(text):

    text = re.sub('\S*@\S*\s?', '', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub("\'", " ", text)
    text = gensim.utils.simple_preprocess(str(text).encode('utf-8'), deacc=True)
    text = remove_stopwords(text)


def main(import_path, export_path):

    global bigram_mod
    global trigram_mod
    global nlp
    global stop_words
    global data_lemmatized
    global id2word

    # NLTK Stop words
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

    X = pd.read_csv(os.path.join(import_path, 'train_X.csv'), usecols=['title'])

    # Convert to list
    data = X.title.values.tolist()

    # Remove Emails
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

    # Remove new line characters
    data = [re.sub('\s+', ' ', sent) for sent in data]

    # Remove distracting single quotes
    data = [re.sub("\'", " ", sent) for sent in data]

    data_words = list(sent_to_words(data))

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)

    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    grid = {}
    grid['Validation_Set'] = {}
    # Topics range
    min_topics = 1
    max_topics = 35
    step_size = 1
    topics_range = range(min_topics, max_topics, step_size)
    # Alpha parameter
    alpha = ['auto']#list(np.arange(0.01, 1, 0.3))
    alpha.append('symmetric')
    alpha.append('asymmetric')
    # Beta parameter
    beta = list(np.arange(0.01, 1, 0.3))
    beta.append('symmetric')
    # Validation sets
    num_of_docs = len(corpus)
    corpus_sets = [corpus]
    corpus_title = ['75% Corpus', '100% Corpus']
    model_results = {
                     'Topics': [],
                     'Coherence': []
                     }
    # Can take a long time to run
    if 1 == 1:
        pbar = tqdm.tqdm(total=len(topics_range))

    # iterate through validation corpuses
        # iterate through number of topics
        for k in topics_range:
            # iterate through alpha values
            # for a in alpha:
                # iterare through beta values
                # for b in beta:
                    # get the coherence score for the given parameters
                cv = compute_coherence_values(corpus=corpus, dictionary=id2word,
                                              k=k, a='auto')
                # Save the model results
                # model_results['Validation_Set'].append(corpus)
                model_results['Topics'].append(k)
                # model_results['Alpha'].append(a)
                # model_results['Beta'].append(b)
                model_results['Coherence'].append(cv)

                pbar.update(1)
    pd.DataFrame(model_results).to_csv(os.path.join(export_path, 'lda_tuning_results.csv'), index=False)
    pbar.close()


if __name__ == '__main__':
    fire.Fire(main(import_path='/Users/aloncohen/Yandex/kaggle_compatition',
                   export_path='/Users/aloncohen/Downloads/'))