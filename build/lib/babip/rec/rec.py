from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import os


def vectorize(data, col_contents):
    stopwords = []

    current_dir = os.path.dirname(os.path.abspath(__file__))
    stopwords_path = os.path.join(current_dir, 'stopwords.txt')

    with open(stopwords_path, 'r', encoding='utf-8') as stopwords_file:
        line = stopwords_file.readline()

        while True:
            line = stopwords_file.readline()
            if not line: break
            stopwords.append(line[:-1])

    stopwords_file.close()

    tfidf = TfidfVectorizer(stop_words=stopwords)
    tfidf_matrix = tfidf.fit_transform(data[col_contents])
    cos = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return tfidf_matrix, cos

def tfidf(data, col_contents):
    return vectorize(data, col_contents)[0]

def cos_similarity(data, col_contents):
    return vectorize(data, col_contents)[1]

def recommend(title, data, col_title, col_contents):
    cos = cos_similarity(data, col_contents)
    title_to_index = dict(zip(data[col_title], data.index))
    idx = title_to_index[title]
    sim_scores = list(enumerate(cos[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    indices = [idx[0] for idx in sim_scores]
    titles = []
    for indice in indices:
        titles.append(data[col_title][indice])
    return titles
