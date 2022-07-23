export MAX_SEQ_LEN=512
export EPOCHS=5
export USE_CUDA=True
export BATCH_SIZE=32
export LR=0.00001
export WEIGHT_DECAY=0.0001
export SAVE_INTERVAL=20
export PRINT_INTERVAL=50
export SAVE_PATH=./output/
export DATA_DIR=/data/private/wanghuadong/liangshihao/QA/data/
export PRETRAINED_PATH=/home/wanghuadong/liangshihao/KEPLER-huggingface/bert-base-multilingual-cased/
export GPU_IDS=6

python main.py \
    --max_seq_len ${MAX_SEQ_LEN} \
    --epochs ${EPOCHS} \
    --use_cuda ${USE_CUDA} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --save_interval ${SAVE_INTERVAL} \
    --print_interval ${PRINT_INTERVAL} \
    --save_path ${SAVE_PATH} \
    --data_dir ${DATA_DIR} \
    --pretrained_path ${PRETRAINED_PATH} \
    --gpu_ids ${GPU_IDS}
