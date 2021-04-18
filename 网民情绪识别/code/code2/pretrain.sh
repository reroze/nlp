python3 run_pretraining.py \
--input_file=./data/pretrained.tfrecord \
--output_dir=./output/my_new_model \
--do_train=True \
--do_eval=True \
--bert_config_file=./albert_tiny_489k/albert_config_tiny.json \
--train_batch_size=128 \
--max_seq_length=128 \
--max_predictions_per_seq=51 \
--num_train_steps=2500 \
--num_warmup_steps=250 \
--learning_rate=0.00005 \
--save_checkpoints_steps=400 \
--init_checkpoint=./albert_tiny_489k/albert_model.ckpt

