from util.tokenize import split_texts_into_words
import numpy as np

def load_word_embedding_glove(path = 'dataset/glove/glove.840B.300d.txt'):
    f = open(path, 'r+')
    word2vec = {}
    lines = f.readlines()
    print('glove中word数量是', len(lines))
    cnt = 0
    for lines in lines:
        vs = lines.split()
        vec = [float(x) for x in vs[-100:]]
        word = ' '.join(vs[:len(vs) - 100])
        if len(vec) != 100: continue
        cnt += 1
        if cnt % 100000 == 0: print("现在处理", cnt)
        word2vec[word] = vec
    embeddings = [list(v) for k, v in word2vec.items()]
    avg_embedding = list(np.mean(embeddings, axis=0))  # 平均embedding
    word2vec['[MEAN]'] = avg_embedding
    return word2vec

def get_word_embedding_dict(texts, dim=64, method='word2vec'):
    print("开始构建word2vec", '句子数', len(texts))
    sentences = split_texts_into_words(texts)

    import multiprocessing
    cores = multiprocessing.cpu_count()
    if method=='word2vec':
        from gensim.models import Word2Vec
        w2v_model = Word2Vec(min_count=1, window=5, vector_size=dim, workers=cores-1) # window=5,前后10个词
    if method == 'fasttext':
        from gensim.models import FastText
        w2v_model = FastText(min_count=1, window=5, vector_size=dim, workers=cores-1)
    print("开始构建word2vec--构建词库")
    w2v_model.build_vocab(sentences, progress_per=100000000)
    # progress_per：指定训练过程中打印日志的间隔。例如，若 progress_per=10000，则每处理 10000 个语料将会打印一次日志。
    print("开始构建word2vec--训练")
    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=100)


    words = list(w2v_model.wv.key_to_index.keys()) # 词汇表
    embeddings = [w2v_model.wv[word] for word in words] # embedding表
    avg_embedding = list(np.mean(embeddings, axis=0)) # 平均embedding

    word2vec = {}
    for i in range(len(words)):
        word2vec[words[i]] = embeddings[i]

    word2vec['[MEAN]'] = avg_embedding

    return word2vec
