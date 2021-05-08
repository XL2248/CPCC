# CPCC
Data for paper "Modeling Bilingual Conversational Characteristics for Neural Chat Translation" (accepted at ACL-IJCNLP 2021 main conference)


Code will be released soon. 


# Introduction

In this paper, we introduced a bilingual Chinese-English chat translation corpus (BMELD).

BMELD is based on the dialogue dataset: [MELD](https://github.com/declare-lab/MELD). It is a multimodal emotionLines dialogue dataset, each utterance of which corresponds to a video, voice, and text, and is annotated with detailed emotion and sentiment. Based on MELD, we firstly crawled the corresponding Chinese translations from [this](https://www.zimutiantang.com/) and then manually post-edited them according to the dialogue history by native Chinese speakers, who are post-graduate students majoring in English. Finally, following the [BConTrasT](https://github.com/Unbabel/BConTrasT) of WMT20 chat translation, we assume 50\% speakers as Chinese speakers to keep data balance for Chinese-English translations and build the Bilingual MELD (BMELD). For the Chinese, we segment the sentence using [Stanford CoreNLP toolkit](https://stanfordnlp.github.io/CoreNLP/index.html).

# BMELD

This file contains the train, dev, and test sets of the BMELD corpus
It is based on the MELD corpus[1] which includes monolingual (i.e. English) dialogs. Each line includes:

  Sr No.	Serial numbers of the utterances mainly for referencing the utterances in case of different versions or multiple copies with different subsets
  
  **Utterance**	Individual utterances from EmotionLines as a string.
  
  Speaker	Name of the speaker associated with the utterance.
  
  Emotion	The emotion (neutral, joy, sadness, anger, surprise, fear, disgust) expressed by the speaker in the utterance.
  
  Sentiment	The sentiment (positive, neutral, negative) expressed by the speaker in the utterance.
  
  Dialogue_ID	The index of the dialogue starting from 0.
  
  Utterance_ID	The index of the particular utterance in the dialogue starting from 0.
  
  Season	The season no. of Friends TV Show to which a particular utterance belongs.
  
  Episode	The episode no. of Friends TV Show in a particular season to which the utterance belongs.
  
  StartTime	The starting time of the utterance in the given episode in the format 'hh:mm:ss,ms'.
  
  EndTime	The ending time of the utterance in the given episode in the format 'hh:mm:ss,ms'.
  
  **Target**	The Chinese translation of the corresponding English Utterance annotated by human.


All MELD dataset[1] was selected and translated into Chinese by native Chinese speakers, who are post-graduate students majoring in English.


**Note:** Following the annotation of BConTrasT[2], we assume 50% speakers speaking Chinese to keep data balance, therefore, the source and target text might be in English or German depending on the role.


# Reference


[1] Soujanya Poria, Devamanyu Hazarika, Navonil Majumder, Gautam Naik, Erik Cambria, and Rada Mihalcea. 2019. MELD: A multimodal multi-party dataset for emotion recognition in conversations. In Proceedings of ACL, pages 527–536.

[2] M. Amin Farajian, Ant ́onio V. Lopes, Andr ́e F. T. Martins, Sameen Maruf, and Gholamreza Haffari. 2020. Findings of the WMT 2020 shared task on chat translation. In Proceedings of WMT, pages 65–75.
