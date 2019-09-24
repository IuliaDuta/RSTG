#!/bin/bash

RAND=$((RANDOM))
DATASET_PATH=./datasets/
SPLIT=valid
CROP=True
CROP_DIM=224
CENTER=True
SMALLER_SIDE=256
MORE_SPACE=False
NUM_MORE_SPACE_CLIPS=1
NUM_TEMP_SAMPLINGS=1
NUM_FRAMES=32
CROP_TEMP=False

args="--split=$SPLIT --dataset_path=$DATASET_PATH --num_frames=$NUM_FRAMES --crop=$CROP --crop_dim=$CROP_DIM --center=$CENTER --smaller_side=$SMALLER_SIDE --num_temporal_samplings=$NUM_TEMP_SAMPLINGS --more_space=$MORE_SPACE --num_more_space_clips=$NUM_MORE_SPACE_CLIPS"

for i in {0..3}
do
    echo "N = $i"
    sleep 2.$[ ( $RANDOM % 100 ) + 1 ]s	
	python -u ./create_datasets/create_smt-smt.py --index=$i --N=4 $args   &
    disown -h
done
