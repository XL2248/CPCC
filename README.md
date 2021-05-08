# CPCC
Data for "Modeling Bilingual Conversational Characteristics for Neural Chat Translation" (ACL2021)


Code will be released soon. 


# Introduction

In this paper, we introduced a bilingual Chinese-English chat translation corpus (BMELD).

BMELD is based on the dialogue dataset: [MELD](https://github.com/declare-lab/MELD). It is a multimodal emotionLines dialogue dataset, each utterance of which corresponds to a video, voice, and text, and is annotated with detailed emotion and sentiment. Based on MELD, we firstly crawled the corresponding Chinese translations from [this](https://www.zimutiantang.com/) and then manually post-edited them according to the dialogue history by native Chinese speakers, who are post-graduate students majoring in English. Finally, following the [setting](https://github.com/Unbabel/BConTrasT) of WMT20 chat translation, we assume 50\% speakers as Chinese speakers to keep data balance for Chinese-English translations and build the Bilingual MELD (BMELD). For the Chinese, we segment the sentence using [Stanford CoreNLP toolkit](https://stanfordnlp.github.io/CoreNLP/index.html).
