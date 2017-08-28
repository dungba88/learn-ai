"""keywords extraction using LDA and NMF"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

N_TOPICS = 5
N_TOP_WORDS = 10

def display_topics(model, feature_names):
    """display generated topics"""
    for idx, topic in enumerate(model.components_):
        topic_words = np.array(feature_names)[np.argsort(topic)][:-N_TOP_WORDS:-1]
        print('[%d]: %s' % (idx, ', '.join(topic_words)))

def extract_lda(comments):
    """extract the topics and keywords from comment using LDA"""
    vectorizer = CountVectorizer(stop_words='english')
    X_train = vectorizer.fit_transform(comments)
    feature_names = vectorizer.get_feature_names()
    model = LatentDirichletAllocation(n_topics=N_TOPICS, learning_method='online', \
                                      max_iter=1500, random_state=0)
    model.fit(X_train)
    print('LDA extractor')
    display_topics(model, feature_names)

def extract_nmf(comments):
    """extract the topics and keywords from comment using NMF"""
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train = vectorizer.fit_transform(comments)
    feature_names = vectorizer.get_feature_names()
    model = NMF(n_components=N_TOPICS, random_state=1, max_iter=1500, init='random')
    model.fit(X_train)
    print('NMF extractor')
    display_topics(model, feature_names)

if __name__ == '__main__':
    COMMENT = 'This hotel is truly a gem. It is a beautifully restored building that provides some \
            of the most unique features that delighted us nearly non-stop. The Music Garden provides \
            a view of the sky yet ingeniously provides protection from the elements. Each room is \
            unique, impeccably decorated, comfortable and equipped with the latest technology. The \
            icing on the cake was the quality of the staff who ALWAYS were polite, pleasant and \
            attentive to our wants and needs. The general manager provided us with his personal \
            attention and has instilled that same level of service in all of the hotel employees. \
            When in Budapest you MUST experience this incredible hotel!'
    extract_lda([COMMENT])
    extract_nmf([COMMENT])
