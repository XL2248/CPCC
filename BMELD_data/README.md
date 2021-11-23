This file contains the train, dev, and test sets of the BMELD corpus

It is based on the MELD [corpus[1]](https://github.com/declare-lab/MELD) which includes monolingual (i.e. English) dialogs. Each line includes:

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


All MELD dataset[1] was selected, translated into Chinese, and then post-edited by native Chinese speakers, who are post-graduate students majoring in English.


**Note:** Following the annotation of BConTrasT[2], they assume 50% speakers speaking Chinese to keep data balance, therefore, the source and target text might be in English or Chinese depending on the role.


# Reference

[1] Soujanya Poria, Devamanyu Hazarika, Navonil Majumder, Gautam Naik, Erik Cambria, and Rada Mihalcea. 2019. MELD: A multimodal multi-party dataset for emotion recognition in conversations. In Proceedings of ACL, pages 527–536.
[2] M. Amin Farajian, Ant ́onio V. Lopes, Andr ́e F. T. Martins, Sameen Maruf, and Gholamreza Haffari. 2020. Findings of the WMT 2020 shared task on chat translation. In Proceedings of WMT, pages 65–75.
