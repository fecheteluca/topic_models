from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re


def read_docs(file_path):
    with open(file_path, 'r') as f:
        text = f.read()
    docs = text.split('<DOC>')
    docs = [doc.split('<TEXT>')[1].split('</TEXT>')[0].strip() for doc in docs if '<TEXT>' in doc]
    return docs


def read_vocab(file_path):
    with open(file_path, 'r') as f:
        vocab = f.read().splitlines()
    return vocab


def clean_text(text):
    # Only include alphabetical words
    return ' '.join(word for word in re.findall(r'\b\w+\b', text) if word.isalpha())


def corpus_matrix(docs, vocab):
    cleaned_docs = [clean_text(doc) for doc in docs]

    vectorizer = CountVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(cleaned_docs)

    n_docs = X.shape[0]
    p_words = X.shape[1]

    D = np.zeros((p_words, n_docs))

    for i in range(n_docs):
        doc_vector = X[i].toarray().flatten()
        N_i = np.sum(doc_vector)
        D[:, i] = doc_vector / N_i if N_i > 0 else 0

    return D

