#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Standard
#SBATCH --gres=gpu:4
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-08:00:00

export CUDA_HOME=/opt/cuda-9.0.176.1/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH


export TMP=/disk/scratch/most_legit/

export DATASET_DIR=${TMP}data
mkdir -p ${TMP}
mkdir -p ${DATASET_DIR}

# echo "Copying dataset."
# rsync -ua /home/${STUDENT_ID}/lanl_earthquake/data/only_train.csv $DATASET_DIR
# rsync -ua /home/${STUDENT_ID}/lanl_earthquake/data/only_val.csv $DATASET_DIR
# echo "DATASET_DIR: $DATASET_DIR"

# Activate the relevant virtual environment:

source /home/${STUDENT_ID}/miniconda3/bin/activate mlp
python scripts/experiments/lstm_experiment.py --data_path /home/${STUDENT_ID}/lanl_earthquake/data \
											 --experiment_name "lstm_fulL_raw3" \
											 --segment_size 150000 --element_size 1000 \
											 --use_gpu "true" --gpu_id "0,1,2,3" \
											 --num_epochs 100 --dropout 0.5 \
											 --learning_rate 0.0003 --batch_size 1 \
											 --num_layers 3 --overlapped_data "false"
