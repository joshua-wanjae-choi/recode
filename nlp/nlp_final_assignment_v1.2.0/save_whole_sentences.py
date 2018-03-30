from konlpy.tag import Mecab
import numpy as np
import pandas as pd
import os
import argparse

def save_whole_sentences_as_morph_txt(name_list):
    mecab = Mecab()
    if os.path.exists('./data/news/whole_articles_morphs.txt'):
        print("File already exists")
    else:
        for name in name_list:
            with open('./data/news/' + name + '.txt', 'r') as f:
                lines = f.read()
                sentences = lines.split('.')
                with open('./data/news/whole_articles_morphs.txt', 'a') as f:
                    f.write(' ')
                    container = []
                    for sentence in sentences :
                        morphs_sentence = mecab.morphs(sentence)
                        container.extend(morphs_sentence)
                    container = str(container)[1:-1]
                    container = container.replace("'", "")
                    container = container.replace(",", " ")
                    container = container.replace("ㆍ", "")
                    container = container.replace("와대", "청와대")
                    f.write(container)
        print("complete saving file")

def save_whole_sentences_as_raw_txt(name_list):
    if os.path.exists('./data/news/whole_articles.txt'):
        print("File already exists")
    else:
        for name in name_list:
            with open('./data/news/' + name + '.txt', 'r') as f:
                lines = f.read()
                with open('./data/news/whole_articles.txt', 'a') as f:
                    f.write(lines)
        print("complete saving file")

def main():
    save_whole_sentences_as_raw_txt(['chosun_full', 'donga_full', 'hani_full', 'joongang_full', 'kh_full'])
    save_whole_sentences_as_morph_txt(['chosun_full', 'donga_full', 'hani_full', 'joongang_full', 'kh_full'])

if __name__=="__main__":
    main()
