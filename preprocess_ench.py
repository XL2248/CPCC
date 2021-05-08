# -*- coding=UTF-8 -*-
#import numpy as np
import pickle, code, re, collections,sys
from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP(r'path_to/stanford-corenlp-full-2018-10-05/', lang='zh')

import csv
ctx_num = int(sys.argv[2])
def replace_abbreviations(text):
    new_text = text
    new_text = pat_letter.sub(' ', text).strip().lower()
    return new_text

def readFileRows(filepath, dimension_size=17):
            #code.interact(local=locals())
    with open(filepath, 'r') as f:
        # file = csv.reader(f)
        # for line in file:
        #     print(line)
        reader = csv.DictReader(f)
        Source = []
        Target = []
        Speaker = []
        Emotion = []
        Sentiment = []
        Dialogue_ID = []
        Utterance_ID = []
        D_ID_list = []
        flag = -1
        for i,row in enumerate(reader):
#            try:
            Source.append(row['Utterance'].decode('utf-8', errors='ignore').encode('utf-8'))
            Speaker.append(row['Speaker'])
            Emotion.append(row['Emotion'])
            sentence = row['Target'].decode("GB18030", errors='ignore').encode('utf-8')
            seg_sentence = nlp.word_tokenize(sentence)
            seg_sent = [token.encode('utf-8') for token in seg_sentence]
            Target.append(" ".join(seg_sent))
            Dialogue_ID.append(row['Dialogue_ID'])
            Utterance_ID.append(row['Utterance_ID'])
            if int(row['Dialogue_ID']) > flag:
                D_ID_list.append(int(row['Dialogue_ID']))
                flag = int(row['Dialogue_ID'])
        English = []
        Chinese = []
        emotion = []
        speaker = []
        switch_context = []
        index = -1
#        tk = MosesTokenizer()
        count = 0
        f_en = open(filepath_w_en, 'w')#, encoding='utf-8-sig')
        f_ch = open(filepath_w_ch, 'w')#, encoding='utf-8-sig')
        f_en_ctx = open(filepath_w_en_ctx, 'w')#, encoding='utf-8-sig')
        f_ch_ctx = open(filepath_w_ch_ctx, 'w')#, encoding='utf-8-sig')
        f_ende_ctx = open(filepath_w_chen_ctx, 'w')#, encoding='utf-8-sig')
        f_perch_ctx = open(filepath_w_chper_ctx, 'w')
        f_peren_ctx = open(filepath_w_enper_ctx, 'w')
        f_emotion = open(filepath_w_emotion, 'w')#, encoding='utf-8-sig')
        f_speaker = open(filepath_w_speaker, 'w')#, encoding='utf-8-sig')
        Dialogue_ID.append(int(Dialogue_ID[-1])+1)
        D_ID_list.append(int(D_ID_list[-1])+1)
        agent, custom = [], []
        for idx in D_ID_list:
            for D_id in Dialogue_ID:
#                code.interact(local=locals())
                if D_id == str(idx): # idx-th dialogue.
                    index += 1
                    English.append(Source[index])
                    Chinese.append(Target[index])
                    emotion.append(Emotion[index])
                    speaker.append(Speaker[index])
                    if index / 2 == 0:
                        switch_context.append(Source[index])
                    else:
                        switch_context.append(Target[index])
                    # dialogue.append(' '.join(tk.tokenize(Utterance[index])))
                else:
                    for k, role in enumerate(speaker): # paired chandler/monica; ross/rachel; phoebe/joey
                        if role not in ["Ross", "Joey", "Rachel"]:
                        #f_q = open(filepath_w_query, 'w')
                        #f_i = open(filepath_w_image, 'w')
                            f_en.write(English[k]+'\n')
                            f_ch.write(Chinese[k]+'\n')
                            f_emotion.write(emotion[k]+'\n')
                            f_speaker.write(speaker[k]+'\n')
                            flag1 = 0
                            chper, enper = [], []
#                        for m in range(k-1, -1, -1):
                        #if k > ctx_num:
                            
                            for m in range(k-1, -1, -1):
                                if speaker[m] == speaker[k]:
                                    chper.append(Chinese[m])
                                    enper.append(English[m])
                                    flag1 += 1
                                if flag1 == ctx_num:
                                    break;

                            if len(chper) == 0:
                               f_perch_ctx.write('pad')                         
                               f_peren_ctx.write('pad')
                            else:
                               chper = list(reversed(chper))
                               enper = list(reversed(enper))
                               f_perch_ctx.write(' ### '.join(chper))
                               f_peren_ctx.write(' @@@ '.join(enper))
                        
                            enctx, chctx, enchctx = [], [], []
                            flag2 = 0
                            for j in range(k-1, -1, -1):
#                        for j in range(begin, k):
                                enctx.append(English[j])
                                chctx.append(Chinese[j])
                                enchctx.append(switch_context[j])
                                flag2 += 1
                                if flag2 == ctx_num:
                                    break;
                            if len(enctx) == 0:
                                f_en_ctx.write('pad')
                                f_ch_ctx.write('pad')
                                f_ende_ctx.write('pad')
                            else:
                                enctx = list(reversed(enctx))
                                chctx = list(reversed(chctx))
                                enchctx = list(reversed(enchctx))
                                f_en_ctx.write(' @@@ '.join(enctx))
                                f_ch_ctx.write(' ### '.join(chctx))
                                f_ende_ctx.write(' @@@ '.join(enchctx))
                            f_en_ctx.write('\n')
                            f_ch_ctx.write('\n')
                            f_ende_ctx.write('\n')
                            f_perch_ctx.write('\n')
                            f_peren_ctx.write('\n')
                         #   f_a.write('\n')
                    #code.interact(local=locals())
                    English = []
                    Chinese = []
                    emotion = []
                    speaker = []
                    switch_context = []
                    #code.interact(local=locals())
        f_en.close()
        f_ch.close()
        f_en_ctx.close()
        f_ch_ctx.close()
        f_emotion.close()
        f_speaker.close()
        f_ende_ctx.close()
        print('count=',count)            # break;

typ = sys.argv[1]
filepath = './'+typ+'_sent_emo.csv'
filepath_w_en = './'+typ+'_en.txt'
filepath_w_ch = './'+typ+'_ch.txt'
filepath_w_en_ctx = './'+typ+'_en_ctx.txt'
filepath_w_ch_ctx = './'+typ+'_ch_ctx.txt'
filepath_w_chen_ctx = './'+ typ+'_chen_ctx.txt'
filepath_w_chper_ctx = './'+ typ+'_chper_ctx.txt'
filepath_w_enper_ctx = './'+ typ+'_enper_ctx.txt'
filepath_w_emotion = './'+typ+'_emotion.txt'
filepath_w_speaker = './'+typ+'_speaker.txt'
readFileRows(filepath)
nlp.close()
