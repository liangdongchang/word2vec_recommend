# -*- coding: utf-8 -*-

"""
@contact: 微信 1257309054
@file: word2vec处理英文句子.py
@time: 2025/1/4 20:34
@author: LDC
"""
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import WordListCorpusReader, stopwords
from nltk.tokenize import word_tokenize


# 可指定stopwords路径
# stopwords=WordListCorpusReader(root=r"C:\Users\ldc\nltk_data\corpora\stopwords", fileids="english")

def get_data():
    """
    获取数据
    :return:
    """

    # 有一个包含图书描述的DataFrame
    data = {
        'book_id': ['travel', 'novel', 'story', 'tale', 'epic', 'detective', 'short stories', 'historical', 'science',
                    'mystery'],
        'description': [
            "This book is about adventure and travel.",
            "A thrilling novel set in a dystopian future.",
            "The story of a young girl's journey through space.",
            "A heartwarming tale of love and loss.",
            "An epic fantasy tale filled with magic and dragons.",
            "A detective story with twists and turns.",
            "A historical novel set in ancient Rome.",
            "A collection of short stories about everyday life.",
            "A science fiction novel with interstellar travel.",
            "A mystery novel set in a small town."
        ]
    }
    df = pd.DataFrame(data)
    return df


# 预处理函数
def preprocess_text(text):
    # 分词
    tokens = word_tokenize(text.lower())
    # 去除停用词
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_tokens)


def train(filename, sentences):
    """
    训练模型
    :param filename: 模型文件名
    :param sentences: 用于训练的词向量
    :return:
    """
    # 预处理所有描述

    '''
    训练Word2Vec模型
    使用Word2Vec训练模型 参数：vector_size: 词向量维度；window: 上下文的宽度，min_count为考虑计算的单词的最低词频阈值
    workers：线程数，epochs：训练次数，negative：负采样，
    sg：sg=1 表示使用 Skip-gram 模型，而 sg=0 表示使用 CBOW（Continuous Bag of Words）模型。
    '''
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=3, workers=4, epochs=7, negative=10, sg=1)
    model.save(filename)  # 模型保存
    return model


def get_book_vector(description, model):
    """
    计算每本图书的向量表示
    将描述中所有词的向量平均作为图书的向量
    :param description:
    :param model:
    :return:
    """
    tokens = word_tokenize(description.lower())  # 分词
    filtered_tokens = []
    # 去除停用词以及非向量词
    # 去除停用词
    stop_words = set(stopwords.words('english'))
    for word in tokens:
        if word.isalnum() and word not in stop_words:
            if word in model.wv:
                filtered_tokens.append(word)

    if not filtered_tokens:
        return np.zeros(model.vector_size)
    book_vector = np.mean([model.wv[word] for word in filtered_tokens], axis=0)
    return book_vector


def get_model(filename):
    # 模型加载
    return Word2Vec.load(filename)


def recommend_books(query_description, df, model, top_n=6):
    """
    推荐图书
    :param query_description: 需要预测的图书描述
    :param df: 数据集
    :param model: 模型
    :param top_n: 推荐的数量
    :return:
    """
    # 获取训练数据的词向量
    book_vectors = df['processed_description'].apply(lambda x: get_book_vector(x, model))
    book_vectors = np.array(list(book_vectors))

    # 获取需要预测文本的词向量
    query_vector = get_book_vector(query_description, model)
    # 逐一计算相似度
    similarities = cosine_similarity([query_vector], book_vectors).flatten()
    # 按照相似度降序排列
    most_similar_indices = similarities.argsort()[::-1][:top_n + 1]
    # 获取对应的图书名称
    return df.iloc[most_similar_indices]


if __name__ == '__main__':
    # 获取数据
    df = get_data()
    # 预处理
    df['processed_description'] = df['description'].apply(preprocess_text)
    # 将所有描述合并为一个长字符串，用于训练Word2Vec
    sentences = df['processed_description'].tolist()

    # 模型训练
    filename = 'book.dat'
    model = train(filename, sentences)

    # 示例：推荐与“A science fiction novel with wilderness  travel.”相似的图书
    query_description = "A science fiction novel with wilderness travel."
    recommended_books = recommend_books(query_description, df, model)
    print(recommended_books[['book_id', 'description']])
