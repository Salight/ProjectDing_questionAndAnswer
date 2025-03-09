# Linux/MacOS
export HF_ENDPOINT=https://hf-mirror.com
python train.py \
    --pretrained_model "uer/t5-base-chinese-cluecorpussmall" \
    --save_dir "checkpoints/DuReaderQG" \
    --train_path "data/DuReaderQG/train.json" \
    --dev_path "data/DuReaderQG/dev.json" \
    --img_log_dir "logs/DuReaderQG" \
    --img_log_name "T5-Base-Chinese" \
    --batch_size 32 \
    --max_source_seq_len 256 \
    --max_target_seq_len 32 \
    --learning_rate 5e-5 \
    --num_train_epochs 50 \
    --logging_steps 10 \
    --valid_steps 500 \
    --device cuda:0