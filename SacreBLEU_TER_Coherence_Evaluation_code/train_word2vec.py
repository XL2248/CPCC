#coding:utf-8
import gensim,os
import codecs,code
from sacremoses import MosesTokenizer
import json

tk = MosesTokenizer()
class MySentences():
    def __init__(self,dirname_list):
        self.dirname_list=dirname_list

    def __iter__(self):
        for line in self.dirname_list:
            pieces = tk.tokenize(line)
            words = [w for w in pieces]
            yield words

all_sentence = []
with open("self-dialogs.json",'r') as load_f: # Dialog file of Taskmaster-1
    load_dict = json.load(load_f)
    for dialog in load_dict:
        for utterances in dialog["utterances"]:
            text = utterances["text"]
            all_sentence.append(text)

sentences = MySentences(all_sentence)
model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=2, iter=100, workers=32)
model.wv.save_word2vec_format('word2vec_dim100.txt',binary=False)