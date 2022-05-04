CONFIG=$1
MODEL_FILE=$2
TASK=$3
python tools/demo_infer.py $CONFIG \
                           $MODEL_FILE \
                           $TASK \
                           ${@:4}