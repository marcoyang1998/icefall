
# Multi-task KD training

## train_multi_KD3.py

For multi-task **unsupervised** KD training, the labels are:
 - Whisper encoder embeddings
 - Speaker embeddings
 - Audio tagging logits

This script extracts the teacher labels in the dataloader, should be faster.

**The latest models are trained with this script**

## train_multi_KD4.py

For multi-task **unsupervised** KD training, the labels are:
 - Whisper encoder embeddings
 - Speaker embeddings
 - Audio tagging logits

This script loads the teacher labels in the training loop, should be **slower**

## train_multi_KD2.py

The former version of train_multi_KD3.py, it contains the mvq loss (not used).
**It is recommended to use train_multi_KD3.py**


## train_multi_KD_with_ASR2.py

AT + SV KD loss and pruned RNNT ASR loss

This script also support fine-tune a model. For example, you can finetune a multi-KD pre-trained model using this script to gain ASR capability while maintaining the good performance of KD training on AT and SV.

# Multi-task supervised training

## train_AT.py

Train an AT model with supervised data

## train_SV.py 

Train a SV model with supervised data

## train_ASR+SV.py

Multi-task supervised training of ASR and SV.

It supports computing both losses for the same sample.

## train_ASR+AT.py

Multi-task supervised training of ASR and AT.

## train_multi_task.py

For multi-task **supervised** training, the labels are:
 - ASR transcripts
 - Speaker labels
 - Audio tagging labels

It also supports **fine-tuning** from a multi-KD trained model.
