import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def calculate_tf(document):
    words = document.split()
    word_count = len(words)
    tf = {}
    for word in set(words):
        tf[word] = words.count(word) / word_count
    return tf

def calculate_idf(documents):
    total_documents = len(documents)
    all_words = ' '.join(documents).split()
    idf = {}
    for word in set(all_words):
        doc_count = sum(1 for doc in documents if word in doc)
        idf[word] = np.log(total_documents / (doc_count + 1))
    return idf

def calculate_tfidf(documents):
    tfidf = []
    tf = [calculate_tf(doc) for doc in documents]
    idf = calculate_idf(documents)
    for doc_tf in tf:
        doc_tfidf = {word: tf_value * idf[word] for word, tf_value in doc_tf.items()}
        tfidf.append(doc_tfidf)
    return tfidf

def normalize_tfidf(tfidf):
    normalized_tfidf = []
    for doc_tfidf in tfidf:
        norm = np.linalg.norm(list(doc_tfidf.values()))
        normalized_tfidf.append({word: value / norm for word, value in doc_tfidf.items()})
    return normalized_tfidf

def calculate_tfidf_sklearn(documents):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    sklearn_tfidf = [{feature_names[j]: tfidf_matrix[0, j]} for j in range(len(feature_names))]
    return sklearn_tfidf

