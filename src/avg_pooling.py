from util.tokenize import split_texts_into_words
import numpy as np


def sentence_embedding_baseline(sentence, embedding_tools):
    word2vec = embedding_tools['word2vec']
    default = word2vec['[MEAN]']
    words = split_texts_into_words([sentence])[0]
    embeddings = []
    for word in words:
        if word in word2vec:
            embeddings.append(word2vec[word])
        else:
            embeddings.append(default)
    if len(embeddings) >= 1:
        embedding = np.mean(embeddings, axis=0)
    else:
        embedding = default
    return embedding


def user_embedding_baseline(sentences, embedding_tools):
    embeddings = []
    for sentence in sentences:
        emb = sentence_embedding_baseline(sentence, embedding_tools)
        embeddings.append(emb)
    embedding = np.mean(embeddings, axis=0)
    return list(embedding)

