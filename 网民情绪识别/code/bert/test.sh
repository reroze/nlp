python3.6 run_classifier.py \
  --task_name=sa \
  --do_predict=true \
  --data_dir=../data \
  --vocab_file=./chinese_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=./chinese_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=../output/trained_model1 \
  --max_seq_length=70 \
  --output_dir=../result/z_trained_model1/

