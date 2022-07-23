dataset=$1

export PRETRAINED_PATH=/home/wanghuadong/liangshihao/KEPLER-huggingface/bert-base-multilingual-cased/
export DATA_ROOT=/data/private/wanghuadong/liangshihao/QA/data/${dataset}/
export JSONL_FILE=PAQ.metadata.jsonl
export PID_ANS_FILE=PAQ.pid.ans.off
export PID_PASSAGE_FILE=psgs_w100.tsv 
export MAX_SEQ_LEN=512

if [ ${dataset} = 'paq' ]
then
    python ./data/preprocess_paq.py ${PRETRAINED_PATH} ${DATA_ROOT} ${JSONL_FILE} ${PID_ANS_FILE} ${PID_PASSAGE_FILE} ${MAX_SEQ_LEN}
elif [ ${dataset} = 'cmrc' ]
then
    python ./data/preprocess_cmrc.py ${PRETRAINED_PATH} ${DATA_ROOT} ${MAX_SEQ_LEN}
elif [ ${dataset} = 'dureader-robust' ]
then
    python ./data/preprocess_dureader.py ${PRETRAINED_PATH} ${DATA_ROOT} ${MAX_SEQ_LEN}
fi