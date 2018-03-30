from konlpy.tag import Mecab
import numpy as np
import pandas as pd
import os
import argparse

from gensim.models import Word2Vec, KeyedVectors
import fasttext
from glove import Corpus, Glove

def save_sentences(name):
    with open('./data/news/' + name + '.txt', 'r') as f:
        lines = f.read()
        sentences = lines.split('.')
        sentences = list(map(lambda x: mecab.morphs(x), sentences[:]))
        sentences = np.array(sentences)
        np.save('./data/news/'+ name + '.npy', sentences)
        print("complete saving " + name)
        return sentences

def load_sentences(name):
    loaded_f = np.load('./data/news/'+ name + '.npy')
    print("complete loading " + name)
    return loaded_f

def read_articles(name_list):
    articles = []
    for news_name in name_list :
        if os.path.exists('./data/news/' + news_name + '.npy'):
            # data = np.array(load_sentences(news_name))
            data = np.array(load_sentences(news_name))
            articles.append(data)
        else :
            data = save_sentences(news_name)
            articles.append(data)
    return articles


def make_word2vec(pretrained, online, size, iter, sg):
    model_list = []
    name_list = []
    if not 'word2vec' in pretrained:
        if online == 'y':
            article_list = ['chosun_full', 'donga_full', 'hani_full', 'joongang_full', 'kh_full']
            articles = read_articles(article_list)
            for idx, sentences in enumerate(articles):
                for idx_size in size:
                    for idx_iter in iter:
                        for idx_sg in sg:
                            print("word2vec_article {:.0f} - size: {:.0f}, iter: {:.0f}, sg: {:.0f}".format(idx, idx_size, idx_iter, idx_sg))
                            model_path = './model/word2vec/word2vec_{:.0f}_{:.0f}_{:.0f}'.format(idx_size, idx_iter, idx_sg)
                            if os.path.exists(model_path):
                                model = Word2Vec.load(model_path)
                                model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
                            else :
                                model = Word2Vec(sentences, min_count=15, size=idx_size, iter=idx_iter, workers=4, sg=idx_sg)
                                model.save(model_path)
                            model_list.append(model)
                            name_list.append('word2vec_{:.0f}_{:.0f}_{:.0f}'.format(idx_size, idx_iter, idx_sg))
        else :
            articles = read_articles(['whole_articles'])
            for sentences in articles:
                for idx_size in size:
                    for idx_iter in iter:
                        for idx_sg in sg:
                            print("word2vec_whole_articles - size: {:.0f}, iter: {:.0f}, sg: {:.0f}".format(idx_size, idx_iter, idx_sg))
                            model_path = './model/word2vec/word2vec_{:.0f}_{:.0f}_{:.0f}'.format(idx_size, idx_iter, idx_sg)
                            model = Word2Vec(sentences, min_count=15, size=idx_size, iter=idx_iter, workers=4, sg=idx_sg)
                            model.save(model_path)
                            model_list.append(model)
                            name_list.append('word2vec_{:.0f}_{:.0f}_{:.0f}'.format(idx_size, idx_iter, idx_sg))
        print("complete training and saving model")
    else :
        for idx_size in size:
            for idx_iter in iter:
                for idx_sg in sg:
                    model_path = './model/word2vec/word2vec_{:.0f}_{:.0f}_{:.0f}'.format(idx_size, idx_iter, idx_sg)
                    model = Word2Vec.load(model_path)
                    print("complete loading model" + model_path)
                    model_list.append(model)
                    name_list.append('word2vec_{:.0f}_{:.0f}_{:.0f}'.format(idx_size, idx_iter, idx_sg))
    return model_list, name_list


def make_glove(pretrained, size, iter, sg):
    model_list = []
    name_list = []
    total_articles = []
    if not 'glove' in pretrained :
        sentences = read_articles(['whole_articles'])
        sentences = list(map(lambda x: x.tolist(), sentences[:]))[0] # list element : np.array -> list
        for idx_size in size:
            for idx_iter in iter:
                    print("glove_whole_articles - size: {:.0f}, iter: {:.0f}".format(idx_size, idx_iter))
                    corpus = Corpus()
                    corpus.fit(sentences, window=10)
                    model = Glove(no_components=idx_size, learning_rate=0.05)
                    model.fit(corpus.matrix, epochs=idx_iter, no_threads=4, verbose=False)
                    model.add_dictionary(corpus.dictionary)
                    model.save('./model/glove/glove_{:.0f}_{:.0f}'.format(idx_size, idx_iter))
                    model_list.append(model)
                    name_list.append('glove_{:.0f}_{:.0f}'.format(idx_size, idx_iter))
        print("complete training and saving model")
    else :
        for idx_size in size:
            for idx_iter in iter:
                    model_path = './model/glove/glove_{:.0f}_{:.0f}'.format(idx_size, idx_iter)
                    model = Glove.load(model_path)
                    print("complete loading model" + model_path)
                    model_list.append(model)
                    name_list.append('glove_{:.0f}_{:.0f}'.format(idx_size, idx_iter))
    return model_list, name_list


def make_fasttext(pretrained, size, iter, sg):
    model_list = []
    name_list = []
    total_articles = []
    if not 'fasttext' in pretrained :
        for idx_size in size:
            for idx_iter in iter:
                for idx_sg in sg:
                    print("fasttext_whole_articles_morphs - size: {:.0f}, iter: {:.0f}, sg: {:.0f}".format(idx_size, idx_iter, idx_sg))
                    model_path = './model/fasttext/fasttext_{:.0f}_{:.0f}_{:.0f}'.format(idx_size, idx_iter, idx_sg)
                    if idx_sg == 0 :
                        model = fasttext.cbow('./data/news/whole_articles_morphs.txt', model_path)
                    elif idx_sg == 1 :
                        model = fasttext.skipgram('./data/news/whole_articles_morphs.txt', model_path)
                    model = KeyedVectors.load_word2vec_format(model_path + '.vec')
                    model_list.append(model)
                    name_list.append('fasttext_{:.0f}_{:.0f}_sg{:.0f}'.format(idx_size, idx_iter, idx_sg))
        print("complete training and saving model")
    else :
        for idx_size in size:
            for idx_iter in iter:
                for idx_sg in sg:
                    model_path = './model/fasttext/fasttext_{:.0f}_{:.0f}_{:.0f}'.format(idx_size, idx_iter, idx_sg)
                    model = KeyedVectors.load_word2vec_format(model_path + '.vec')
                    print("complete loading model" + model_path)
                    model_list.append(model)
                    name_list.append('fasttext_{:.0f}_{:.0f}_sg{:.0f}'.format(idx_size, idx_iter, idx_sg))
    return model_list, name_list


def get_most_similar(model_list, name_list, words, var_name):
    similar_list = []
    for model in model_list :
        for word in words:
            if "glove" in name_list[0]:
                similar_list.append(model.most_similar(word, number=2)[0][0])
            else :
                similar_list.append(model.wv.most_similar(positive=[word], topn=1)[0][0])
    similar_list = np.array(similar_list).reshape(-1,len(words))
    df = pd.DataFrame(similar_list, columns=words, index=name_list)
    df.to_csv('./output/'+ var_name +'.csv', encoding='utf-8')
    # print(df)
    print("complete saving most_similar output")

def get_word_analogy(model_list, name_list, candi_word1, candi_word2, candi_word3, var_name):
    analogy_list = []
    candidate_words = []
    FLAG_get_candidate_words=0
    for model in model_list:
        for cand1, cand2, cand3 in zip(candi_word1, candi_word2, candi_word3):
            if "glove" in name_list[0]:
                analogy_list.append(get_glove_word_analogy(word=cand1, positive=cand2, negative=cand3, model=model, number=1)[0][0])
            else :
                analogy_list.append(model.wv.most_similar_cosmul(positive=[cand1, cand2], negative=[cand3])[0][0])
            if FLAG_get_candidate_words == 0:
                candidate_words.append('%s + %s - %s' % (cand1, cand2, cand3))
        if FLAG_get_candidate_words == 0:
            FLAG_get_candidate_words = 1

    analogy_list = np.array(analogy_list).reshape(-1, len(candidate_words))
    df = pd.DataFrame(analogy_list, columns=candidate_words, index=name_list)
    df.to_csv('./output/'+ var_name +'.csv', encoding='utf-8')
    # print(df)
    print("complete saving word_analogy output")

def get_glove_word_analogy(word, model, positive, negative, number=5):
    dictionary = model.dictionary
    word_vectors = model.word_vectors
    inverse_dictionary = model.inverse_dictionary
    number += 1

    try:
        word_idx = dictionary[word]
    except KeyError:
        raise Exception('Word not in dictionary')

    word_vec = word_vectors[word_idx]
    word_vec += word_vectors[dictionary[positive]]
    word_vec -= word_vectors[dictionary[negative]]

    dst = (np.dot(word_vectors, word_vec)
    / np.linalg.norm(word_vectors, axis=1)
    / np.linalg.norm(word_vec))
    word_ids = np.argsort(-dst)
    return [(inverse_dictionary[x], dst[x]) for x in word_ids[0:number]][1:]


def main() :
    parser = argparse.ArgumentParser(description='This code is transforming news_data to vector-form')
    parser.add_argument('--pretrained', type=str, nargs='+', default='none',\
                        help="if you have already pretrained model, insert model name e.g.--pretrain word2vec \n \
                              pretraned model : word2vec, glove, fasttext")
    parser.add_argument('--online', type=str, default='n',\
                        help="if you want to use online-learning, insert 'y' or you don't, insert 'n' e.g.--online n \
                              *you can only train word2vec using online-learning")
    parser.add_argument('--size', nargs='+', type=int, default=[100],\
                        help="list of embedding sizes. e.g.--size 100 300 500")
    parser.add_argument('--iter', type=int, nargs='+', default=[10],\
                        help="list of # iteration. e.g.--iter 10 20")
    parser.add_argument('--sg', type=int, nargs='+', default=[0],\
                        help= "list of model type. e.g.--sg 0 or --sg 1 or --sg 0 1 \n \
                               CBOW has sg value 0, otherwise like skip-gram has sg value 1")


    args = parser.parse_args()
    pretrained = args.pretrained
    size = args.size
    iter = args.iter
    sg = args.sg
    online = args.online

    print("Check pretrained model: ",pretrained)
    global whole_word
    # whole_word = ['박근혜', '오바마', '김정은', '아베', '청와대', '백악관', '이', '가', '은', '는', '었', '지만', '것', '구나', '않', '안', '못', '있', '없', '서울', '일본', '한국','미국', '김무성', '민주주의', '보수', '북한']
    test_words = ['박근혜', '오바마', '김정은', '아베', '청와대', '백악관', '이', '가', '은', '는', '었', '지만', '것', '구나', '않', '안', '못', '있', '없']
    # missing word: 새누리당, 새정치, 최순실

    candi_word1 = ['서울', '서울', '서울', '박근혜', '박근혜', '박근혜', '청와대', '민주주의']
    candi_word2 = ['일본', '미국', '북한', '일본', '미국', '북한', '미국', '북한']
    candi_word3 = ['한국', '한국', '한국', '한국', '한국', '한국', '한국', '한국']
    # missing word: 새누리당, 새정치


    global mecab
    mecab = Mecab()
    word2vec_models, word2vec_names = make_word2vec(pretrained, online, size, iter, sg)
    get_most_similar(word2vec_models, word2vec_names, test_words, var_name="word2vec_most_similar")
    get_word_analogy(word2vec_models, word2vec_names, candi_word1, candi_word2, candi_word3, var_name="word2vec_word_analogy")

    glove_models, glove_names = make_glove(pretrained, size, iter, sg)
    get_most_similar(glove_models, glove_names, test_words, var_name="glove_most_similar")
    get_word_analogy(glove_models, glove_names, candi_word1, candi_word2, candi_word3, var_name="glove_word_analogy")

    fasttext_models, fasttext_names = make_fasttext(pretrained, size, iter, sg)
    get_most_similar(fasttext_models, fasttext_names, test_words, var_name="fasttext_most_similar")
    get_word_analogy(fasttext_models, fasttext_names, candi_word1, candi_word2, candi_word3, var_name="fasttext_word_analogy")


if __name__=="__main__":
    main()
