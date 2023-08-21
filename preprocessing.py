import numpy as np
import re
import string
import inflect
from num2words import num2words


def preprocessing(dir, dataset, words_percent, docs_percent):
    stopwords1 = np.loadtxt(dir + '/Recover/code/stopwords.txt', dtype=str)
    stopwords2 = np.loadtxt(dir + '/TSVD/code/stopwords.txt', dtype=str)
    stopwords = np.unique(np.concatenate((stopwords1, stopwords2)))
    np.chdir(dir + '/real_data')
    vocab = np.loadtxt(dataset + '.vocab.txt', dtype=str)
    data = np.loadtxt(dir + '/real_data/' + dataset + '.txt', dtype=float, delimiter=' ', ndmin=2)

    del_stopwords = np.zeros(len(vocab))
    t = 1
    for i in range(len(vocab)):
        if vocab[i] not in stopwords:
            del_stopwords[i] = t
            t += 1
    vocab = vocab[del_stopwords != 0]
    data = data[del_stopwords[data[:, 1].astype(int)] != 0, :]
    data[:, 1] = del_stopwords[data[:, 1].astype(int)]
    n = np.max(data[:, 0])
    p = np.max(data[:, 1])
    D = np.zeros((p, n))
    for t in range(data.shape[0]):
        D[int(data[t, 1]) - 1, int(data[t, 0]) - 1] = data[t, 2]

    D_rowsum = np.sum(D, axis=1)
    del_low_words = np.zeros(D.shape[0])
    threshold = np.quantile(D_rowsum, 1 - words_percent)
    t = 1
    for i in range(D.shape[0]):
        if D_rowsum[i] >= threshold:
            del_low_words[i] = t
            t += 1
    vocab = vocab[del_low_words != 0]
    D = D[del_low_words != 0, :]

    D_colsum = np.sum(D, axis=0)
    threshold = np.quantile(D_colsum, 1 - docs_percent)
    D = D[:, D_colsum >= threshold]

    with open("prepro_" + str(words_percent) + "_" + str(docs_percent) + "_" + dataset + ".txt", 'w') as fileConn:
        pass
    with open("prepro_" + str(words_percent) + "_" + str(docs_percent) + "_" + dataset + ".txt", 'a') as fileConn:
        for i in range(D.shape[0]):
            for j in range(D.shape[1]):
                if D[i, j] > 0:
                    fileConn.write(str(j + 1) + " " + str(i + 1) + " " + str(D[i, j]) + "\n")

    with open("prepro_" + str(words_percent) + "_" + str(docs_percent) + "_" + dataset + ".vocab.txt", 'w') as fileConn:
        pass
    with open("prepro_" + str(words_percent) + "_" + str(docs_percent) + "_" + dataset + ".vocab.txt", 'a') as fileConn:
        for i in range(len(vocab)):
            fileConn.write(vocab[i] + "\n")


def extract_texts(filename):
    with open(filename, 'r') as file:
        data = file.read()

    text_list = re.findall(r'<TEXT>\n(.*?)\n </TEXT>', data, re.DOTALL)

    word_lists = []

    p = inflect.engine()

    for text in text_list[:10]:
        words = text.split()
        for i, word in enumerate(words):
            # Remove punctuation
            word = word.translate(str.maketrans('', '', string.punctuation))

            # Convert numbers to words
            if word.isdigit():
                words[i] = num2words(word)
            elif word.replace('.', '', 1).isdigit():  # check for float
                words[i] = num2words(float(word))
            elif word in p.abbreviations:  # check for abbreviations
                words[i] = p.abbreviations[word]

        word_lists.append(words)

    return word_lists


filename = 'data/press_data.txt'
word_lists = extract_texts(filename)

for i, words in enumerate(word_lists):
    print(f'Text {i + 1}:')
    print(words)
    print()
