RAND=$((RANDOM))


MODEL_DIR='./checkpoints/'$1
LOG_NAME=$MODEL_DIR'/log_'$RAND
CONFIG_FILE='./configs/name_of_config_file.yaml'
mkdir $MODEL_DIR


args="--model_dir=$MODEL_DIR --rand_no=$RAND --config_file=$CONFIG_FILE"


CUDA_VISIBLE_DEVICES=0 python -u train.py $args  |& tee $LOG_NAME
