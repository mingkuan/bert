#!/usr/bin/env bash

## testing original BERT classification task
#export BERT_BASE_DIR=./pretrained/uncased_L-12_H-768_A-12
#export GLUE_DIR=./GLUE/glue_data/
#
#python run_classifier.py \
#  --task_name=MRPC \
#  --do_train=true \
#  --do_eval=true \
#  --data_dir=$GLUE_DIR/MRPC \
#  --vocab_file=$BERT_BASE_DIR/vocab.txt \
#  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
#  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
#  --max_seq_length=128 \
#  --train_batch_size=32 \
#  --learning_rate=2e-5 \
#  --num_train_epochs=3.0 \
#  --output_dir=./mrpc_output/

# testing NER task
#
## testing classification task
#export BERT_BASE_DIR=./pretrained/uncased_L-12_H-768_A-12
#export DATA_DIR=./mysample/eCommerceCatData/
#
#python my_classifier.py \
#  --task_name=eBayCat \
#  --do_train=true \
#  --do_eval=true \
#  --data_dir=$DATA_DIR \
#  --vocab_file=$BERT_BASE_DIR/vocab.txt \
#  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
#  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
#  --max_seq_length=64 \
#  --train_batch_size=32 \
#  --learning_rate=2e-5 \
#  --num_train_epochs=1.0 \
#  --output_dir=./ebaycat_output/
#
## testing relevance tasks


# testing full categorization task
export BERT_BASE_DIR=./pretrained/uncased_L-12_H-768_A-12
export DATA_DIR=./mysample/fullCategoryClassifyData/

python my_classifier.py \
  --task_name=fullcat \
  --do_train=true \
  --do_eval=true \
  --data_dir=$DATA_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=20 \
  --train_batch_size=32 \
  --save_checkpoints_steps=100000 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=./fullcat_output/
