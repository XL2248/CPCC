#coding=utf-8
#import numpy as np
import pickle, code, re, collections, sys

import csv
ctx_num = int(sys.argv[2])

def readFileRows(filepath, dimension_size=17):
            #code.interact(local=locals())
    with open(filepath, 'r', encoding='utf-8-sig') as f:
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
        for row in reader:
            Source.append(row['Source'])
            Speaker.append(row['Speaker'])
            Emotion.append(row['Emotion'])
            Target.append(row['Target'])
            Dialogue_ID.append(row['Dialogue_ID'])
            Utterance_ID.append(row['Utterance_ID'])
        English = []
        German = []
        emotion = []
        switch_context = []
        index = -1
#        tk = MosesTokenizer()
        count = 0
        f_en = open(filepath_w_en, 'w', encoding='utf-8-sig')
        f_de = open(filepath_w_de, 'w', encoding='utf-8-sig')
        f_en_ctx = open(filepath_w_en_ctx, 'w', encoding='utf-8-sig')
        f_de_ctx = open(filepath_w_de_ctx, 'w', encoding='utf-8-sig')
        f_emotion = open(filepath_w_emotion, 'w', encoding='utf-8-sig')
        f_ende_ctx = open(filepath_w_ende_ctx, 'w', encoding='utf-8-sig')
        f_peren_ctx = open(filepath_w_enper_ctx, 'w', encoding='utf-8-sig')
        f_perde_ctx = open(filepath_w_deper_ctx, 'w',encoding='utf-8-sig')
        Dialogue_ID.append(len(set(Dialogue_ID))+1)
        Agent = []
        for idx in range(len(set(Dialogue_ID))):
            for D_id in Dialogue_ID:
#                code.interact(local=locals())
                if D_id == str(idx): # idx-th dialogue.
                    index += 1
                    switch_context.append(Source[index])
                    if Speaker[index] == "agent":
                        English.append(Source[index])
                        German.append(Target[index])
                        Agent.append("agent")
                    else:
                        German.append(Source[index])
                        English.append(Target[index])
                        Agent.append("customer")
                    emotion.append(Emotion[index])
                else:
                    # if len(dialogue)
                    #code.interact(local=locals())
                    for k, role in enumerate(Agent):
                        if role == "agent":
#                    for k in range(len(English)):
                        #f_q = open(filepath_w_query, 'w')
                        #f_i = open(filepath_w_image, 'w')
                            f_en.write(English[k]+'\n')
                            f_de.write(German[k]+'\n')
                            f_emotion.write(emotion[k]+'\n')
                            flag1 = 0
                            deper, enper = [], []
                            for m in range(k-1, -1, -1):
                            #for m in range(0, k):
                                if Agent[m] == Agent[k]:
                                    enper.append(English[m])
                                    deper.append(German[m])
                                    flag1 += 1
                                if flag1 == ctx_num:
                                    break;
                            if len(enper) == 0:
                                f_peren_ctx.write('pad')
                                f_perde_ctx.write('pad')
                            else:
                                enper = list(reversed(enper))
                                deper = list(reversed(deper))
                                f_peren_ctx.write(' @@@ '.join(enper))
                                f_perde_ctx.write(' @@@ '.join(deper))#ctx_num = 10

                            enctx, dectx, endectx = [], [], []
                            flag2 = 0
                            for j in range(k-1, -1, -1):
#                            for j in range(0, k):
                                dectx.append(German[j])
                                enctx.append(English[j])
                                endectx.append(switch_context[j])
                                flag2 += 1
                                if flag2 == ctx_num:
                                    break;

                            if len(enctx) == 0:
                                f_en_ctx.write('pad')
                                f_de_ctx.write('pad')
                                f_ende_ctx.write('pad')
                            else:
                                enctx = list(reversed(enctx))
                                dectx = list(reversed(dectx))
                                endectx = list(reversed(endectx))
                                f_en_ctx.write(' @@@ '.join(enctx))
                                f_de_ctx.write(' @@@ '.join(dectx))
                                f_ende_ctx.write(' @@@ '.join(endectx))
                            f_en_ctx.write('\n')
                            f_de_ctx.write('\n')
                            f_ende_ctx.write('\n')
                            f_perde_ctx.write('\n')
                            f_peren_ctx.write('\n')                            
                    English = []
                    German = []
                    emotion = []
                    Agent = []
                    switch_context = []
                    #code.interact(local=locals())
        f_en.close()
        f_de.close()
        f_en_ctx.close()
        f_de_ctx.close()
        f_emotion.close()
        f_ende_ctx.close()
        print('count=',count)            # break;


typ = 'dev'
typ = sys.argv[1]
filepath = 'en_de_'+typ+'_emotion.csv'
filepath_w_en = typ+'_en.txt'
filepath_w_de = typ+'_de.txt'
filepath_w_en_ctx = typ+'_en_ctx.txt'
filepath_w_de_ctx = typ+'_de_ctx.txt'
filepath_w_ende_ctx = typ+'_ende_ctx.txt'
filepath_w_enper_ctx = typ+'_enper_ctx.txt'
filepath_w_deper_ctx = typ+'_deper_ctx.txt'
filepath_w_emotion = typ+'_emotion.txt'
readFileRows(filepath)
