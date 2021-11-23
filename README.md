# CPCC
Data and codes for the ACL-IJCNLP-2021 main conference paper [Modeling Bilingual Conversational Characteristics for Neural Chat Translation](https://aclanthology.org/2021.acl-long.444/).

# Introduction

In this paper, we introduced a bilingual Chinese-English chat translation corpus (BMELD).

## BMELD

This file contains the train, dev, and test sets of the BMELD corpus
It is based on the MELD corpus[1] which includes monolingual (i.e. English) dialogs. 

Each line includes:

  Sr No.  Serial numbers of the utterances mainly for referencing the utterances in case of different versions or multiple copies with different subsets
  
  **Utterance**:  Individual utterances from EmotionLines as a string.
  
  Speaker:  Name of the speaker associated with the utterance.
  
  Emotion:  The emotion (neutral, joy, sadness, anger, surprise, fear, disgust) expressed by the speaker in the utterance.
  
  Sentiment:  The sentiment (positive, neutral, negative) expressed by the speaker in the utterance.
  
  Dialogue_ID:  The index of the dialogue starting from 0.
  
  Utterance_ID: The index of the particular utterance in the dialogue starting from 0.
  
  Season: The season no. of Friends TV Show to which a particular utterance belongs.
  
  Episode:  The episode no. of Friends TV Show in a particular season to which the utterance belongs.
  
  StartTime:  The starting time of the utterance in the given episode in the format 'hh:mm:ss,ms'.
  
  EndTime:  The ending time of the utterance in the given episode in the format 'hh:mm:ss,ms'.
  
  **Target**: The Chinese translation of the corresponding English Utterance annotated by human.


**Note:** Following the annotation of BConTrasT[2], we assume 50% speakers speaking Chinese to keep data balance, therefore, the source and target text might be in English or Chinese depending on the role.

BMELD is based on the dialogue dataset: [MELD](https://github.com/declare-lab/MELD). It is a multimodal emotionLines dialogue dataset, each utterance of which corresponds to a video, voice, and text, and is annotated with detailed emotion and sentiment. Based on MELD, we firstly crawled the corresponding Chinese translations from [this](https://www.zimutiantang.com/) and then manually post-edited them according to the dialogue history by native Chinese speakers, who are post-graduate students majoring in English. Finally, following the [BConTrasT](https://github.com/Unbabel/BConTrasT) of WMT20 chat translation, we assume 50\% speakers as Chinese speakers to keep data balance for Chinese-English translations and build the Bilingual MELD (BMELD). For the Chinese, we segment the sentence using [Stanford CoreNLP toolkit](https://stanfordnlp.github.io/CoreNLP/index.html).


# Reference


[1] Soujanya Poria, Devamanyu Hazarika, Navonil Majumder, Gautam Naik, Erik Cambria, and Rada Mihalcea. 2019. MELD: A multimodal multi-party dataset for emotion recognition in conversations. In Proceedings of ACL, pages 527–536.

[2] M. Amin Farajian, Ant ́onio V. Lopes, Andr ́e F. T. Martins, Sameen Maruf, and Gholamreza Haffari. 2020. Findings of the WMT 2020 shared task on chat translation. In Proceedings of WMT, pages 65–75.


# Training (Take En->De as an example)
Our code is basically based on the publicly available toolkit: [THUMT-Tensorflow](https://github.com/THUNLP-MT/THUMT) (our python version 3.6).
The following steps are training our model and then test its performance in terms of BLEU, TER, and Sentence Similarity.

## Data Preprocessing
Please refer to the "data_preprocess_code" file.

## Two-stage Training

+ The first stage

```
1) bash train_ende_base1.sh # Suppose the generated checkpoint file is located in path1
```
+ The second stage (i.e., fine-tuning on the chat translation data)

```
2) bash train_ende_base2.sh # Here, set the training_step=1; Suppose the generated checkpoint file is located in path2
3) python thumt-code1/thumt/scripts/combine_add.py --model path2 --part path1 --output path3  # copy the weight of the first stage to the second stage.
4) bash train_ende_base2.sh # Here, set the --output=path3 and the training_step=22,000; Suppose the generated checkpoint file is path4
```
+ Test by multi-blue.perl

```
5) bash test_en_de2.sh # set the checkpoint file path to path4 in this script. # Suppose the predicted file is located in path5 at checkpoint step xxxxx
```
+ Test by SacreBLEU and TER
Required TER: v0.7.25; Sacre-BLEU: version.1.4.13 (BLEU+case.mixed+numrefs.1+smooth.exp+tok.13a+version.1.4.13)

```
6) python SacreBLEU_TER_Coherence_Evaluation_code/cal_bleu_ter4ende.py # Please correctly set the golden file and predicted file in this file and in sacrebleu_ende.py, respectively.
```

+ Coherence Evaluation by Sentence Similarity
Required: gensim; MosesTokenizer

```
7) python SacreBLEU_TER_Coherence_Evaluation_code/train_word2vec.py # firstly downloading the corpus in [2] and then training the word2vec.
8) python SacreBLEU_TER_Coherence_Evaluation_code/eval_coherence.py # putting the file containing three precoding utterances and the predicted file in corresponding location and then running it.
```

# Citation
If you find this project helps, please cite our paper :)

```
@inproceedings{liang-etal-2021-modeling,
    title = "Modeling Bilingual Conversational Characteristics for Neural Chat Translation",
    author = "Liang, Yunlong  and
      Meng, Fandong  and
      Chen, Yufeng  and
      Xu, Jinan  and
      Zhou, Jie",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.444",
    doi = "10.18653/v1/2021.acl-long.444",
    pages = "5711--5724",
}
```
