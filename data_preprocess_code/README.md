# preprocessing En-De 
The En-De data is located in "BConTrasT_data".
python version 3.6
python preprocess_ende.py train 3  # 3 precoding utterances
python preprocess_ende.py test 3
python preprocess_ende.py dev 3



# preprocessing En-Ch
Note that the stanford-corenlp-full-2018-10-05 file is necessary to segment Chinese sentence.
The En-Ch data is located in "BMELD".
python version 2.7
python preprocess_ench.py train 3  # 3 precoding utterances
python preprocess_ench.py test 3
python preprocess_ench.py dev 3


Then we apply BPE to generate the corresponding files and then start to the two-stage training.

Done!