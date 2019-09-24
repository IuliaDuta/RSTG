RAND=$((RANDOM))


MODEL_DIR='/models/graph_models/models_5_digits_sincron_noiseless/'$1
LOG_NAME=$MODEL_DIR'/log_'$RAND
CONFIG_FILE='./configs/mnist_config.yaml'
mkdir $MODEL_DIR


args="--model_dir=$MODEL_DIR --rand_no=$RAND --config_file=$CONFIG_FILE"


CUDA_VISIBLE_DEVICES=0 python -u train.py $args  |& tee $LOG_NAME
