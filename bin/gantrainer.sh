#!/bin/bash
#DSUB -n gantrainer
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0258_tmp
#DSUB -R "cpu=24;gpu=1;mem=50000"
#DSUB -oo logs/gantrainer.out.%J
#DSUB -eo logs/gantrainer.err.%J

source /home/HPCBase/tools/module-5.2.0/init/profile.sh
module use /home/HPCBase/modulefiles/
module load compilers/cuda/11.8.0
module load libs/cudnn/8.6.0_cuda11
module load libs/nccl/2.18.3_cuda11
module load compilers/gcc/12.3.0

JOB_PATH="/home/share/huadjyin/home/s_huluni/fengboyu/bio_tools/CATree/sctGAN"
cd ${JOB_PATH}
python -u GANTrain.py --sc_dir /home/share/huadjyin/home/s_huluni/mashubao/scSRT/Dataset/SC/MERFISH_read/sc_downsampled.h5ad \
                    --st_dir /home/share/huadjyin/home/s_huluni/mashubao/scSRT/spatialID-main/dataset/MERFISH/mouse1_sample1.h5ad