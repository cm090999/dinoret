#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=18
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpumem:20g
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=6000
#SBATCH --tmp=4000 # per node!!
#SBATCH --job-name=dino_run
#SBATCH --output=dino_run.out
#SBATCH --error=dino_run.err
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END


# Load the required modules
module load gcc/8.2.0 python_gpu/3.11.2
module load eth_proxy

source .env//bin/activate

echo "GPUs allocated by SLURM: ${CUDA_VISIBLE_DEVICES}"

FREE_PORT=$(python -c 'import socket; sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM); sock.bind(("", 0)); print(sock.getsockname()[1]); sock.close()')
echo "Free Port: ${FREE_PORT}"

num_tasks=$SLURM_NTASKS
echo "Number of workers: ${num_tasks}"

#This is blocking don't uncomment!!!

test_path=/add/your/path
rm -rf $test_path

torchrun --nproc_per_node=1 --master_port=$FREE_PORT main_finetune.py \
 --batch_size 32 --world_size 1 --model dinov2_vitb14 \
 --epochs 100 --warmup_epochs 5 --blr 2e-3 --weight_decay 0 \
 --drop_path 0.0 --nb_classes 5 --num_workers $num_tasks  \
 --data_path /add/your/path \
 --task $test_path \
 --input_size 224 \
 --RandomResizeCrop_upperBound 1.2 --RandomResizeCrop_lowerBound 0.8 \
 --RandomRotation_degrees 5 \
 --loss_weights True --uniform_class_prob_training False --smoothing 0.0 \
 --mixup 0.0 --cutmix 0.0 --mixup_prob 0.0 --mixup_switch_prob 0.0 --mixup_mode batch \
 --preprocessing_CropRoundImage False --preprocessing_CLAHETransform False --preprocessing_MedianFilterTransform False \
 --color_jitter_param_brightness 0.2 --color_jitter_param_contrast 0.2 --color_jitter_param_saturation 0.1 --color_jitter_param_hue 0.1 \
 --fix_backbone True --n_classification_heads 1
