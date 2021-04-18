#!/usr/bin/env bash

BERT_BASE_DIR=./albert_tiny_489k
python3 create_pretraining_data.py --do_whole_word_mask=True --input_file=data/pretrained_data.txt \
--output_file=data/new_pretrained.tfrecord --vocab_file=$BERT_BASE_DIR/vocab.txt --do_lower_case=True \
--max_seq_length=128 --max_predictions_per_seq=51 --masked_lm_prob=0.10
