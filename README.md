# CPCC
Data for "Modeling Bilingual Conversational Characteristics for Neural Chat Translation" (ACL2021)
Code will be released soon. 

## Introduction

In this paper, we introduced a bilingual Chinese-English chat translation corpus.

Based on the dialogue dataset in the MELD (originally in English)~\cite{poria-etal-2019-meld}\footnote{The MELD is a multimodal emotionLines dialogue dataset, each utterance of which corresponds to a video, voice, and text, and is annotated with detailed emotion and sentiment.}, we firstly crawled the corresponding Chinese translations from this\footnote{https://www.zimutiantang.com/} and then manually post-edited them according to the dialogue history by native Chinese speakers, who are post-graduate students majoring in English. Finally, following~\cite{farajian-etal-2020-findings}, we assume 50\% speakers as Chinese speakers to keep data balance for Ch$\Rightarrow$En translations and build the \underline{b}ilingual MELD (BMELD). For the Chinese, we segment the sentence using [Stanford CoreNLP toolkit](https://stanfordnlp.github.io/CoreNLP/index.html).
