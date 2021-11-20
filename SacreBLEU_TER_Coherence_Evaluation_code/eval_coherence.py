#coding=utf-8
import code,os
from sacremoses import MosesTokenizer
import numpy as np

tk = MosesTokenizer()
def bit_product_sum(x, y):
    return sum([item[0] * item[1] for item in zip(x, y)])


def cosine_similarity(x, y, norm=True):

    if len(x) != len(y): 
        return float(0)
    zero_list = [0] * len(x)
    if x == zero_list or y == zero_list:
        return float(1) if x == y else float(0)

    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

    return 0.5 * cos + 0.5 if norm else cos 

def read_embedding(word_list, golden_list, file_path="default_embedding_path", dimension_size = 100, uniform_scale = 0.25):
    word2embed = {}
    with open(file_path, 'r') as fopen:
        for line in fopen:
            w = line.strip().split()
            word2embed[' '.join(w[:-dimension_size])] = w[-dimension_size:]
    word_vectors = []

    ground_truth = []
    cw = set()
    g_set = set()
    for line in golden_list:
        sentence_emb = 0
        for word in line:
            g_set.add(word)
            if word in word2embed:
                cw.add(word)
                sentence_emb += np.array(word2embed[word], dtype=np.float32)
            else:
                sentence_emb += np.random.uniform(-uniform_scale, uniform_scale, dimension_size)
        ground_truth.append(sentence_emb/len(line)) # sum sentence embedding

    c = 0
    predict = []
    for line in word_list:
        sentence_emb = 0
        for word in line:
            if word in word2embed:
                c += 1
                sentence_emb += np.array(word2embed[word], dtype=np.float32)
            else:
                sentence_emb +=np.random.uniform(-uniform_scale, uniform_scale, dimension_size)
        predict.append(sentence_emb/len(line))

    sim = 0
    for g, p in zip(ground_truth, predict):
        sim += cosine_similarity(g.tolist(), p.tolist())
    return sim/len(ground_truth)

if __name__ == '__main__':
    first_golden_list = [] # last
    second_golden_list = []
    third_gold_list = []
    predict_list = []
    pre_target_file = "test_ctx.tok.en" # containing the three precoding utterances
    with open(pre_target_file, 'r') as fopen:
        for line in fopen:
            tmp = line.strip().split("@ @ @")
            if len(tmp) == 3:
                 first_golden_list.append([w for w in tmp[2].strip().split()])
                 second_golden_list.append([w for w in tmp[1].strip().split()])
                 third_gold_list.append([w for w in tmp[0].strip().split()])
            if len(tmp) == 2:
                 first_golden_list.append([w for w in tmp[1].strip().split()])
                 second_golden_list.append([w for w in tmp[0].strip().split()])
                 third_gold_list.append([""])
            if len(tmp) == 1:
                 first_golden_list.append([w for w in tmp[0].strip().split()])
                 second_golden_list.append([""])
                 third_gold_list.append([""])
            if len(tmp) == 0:
                 first_golden_list.append([""])
                 second_golden_list.append([""])
                 third_gold_list.append([""])

    for emb_file in [100]: #
#        code.interact(local=locals())
        print("embedding_iter:", emb_file)
        for idx in [1, 376201, 377251, 376601, 377302, 205413, 205012]: # predicted file at each checkpoint step.
            predict_list = []
            with open('translation/deen_output/test.out.en.delbpe.'+str(idx), 'r') as fopen:
                for line in fopen:
                    tmp = [w for w in line.strip().split()]
                    predict_list.append(tmp)
            print("previous 1-th:", read_embedding(predict_list, first_golden_list, "word2vec_dim"+str(emb_file)+".txt", 100))
            print("previous 2-th:", read_embedding(predict_list, second_golden_list, "word2vec_dim"+str(emb_file)+".txt", 100))
            print("previous 3-th:", read_embedding(predict_list, third_gold_list, "word2vec_dim"+str(emb_file)+".txt", 100))



