CONFIG=$1
MODEL_FILE=$2
python tools/demo_infer.py $CONFIG \
                           $MODEL_FILE \
                           ${@:3}