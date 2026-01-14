#!/bin/bash
###
 # @Author: WangHongbinary
 # @E-mail: hbwang22@gmail.com
 # @Date: 2025-12-10 10:43:41
 # @LastEditTime: 2025-12-23 14:21:33
 # @Description: EEGNet | DeepConvNet | ShallowConvNet | BrainModule | CNN-BiGRU | Conformer | DBConformer
### 

seeds=(2024 2025 2026 2027 2028 2029)
gpus=(1 2 3 4 5 6)

backbones=('EEGNet' 'DeepConvNet' 'ShallowConvNet' 'BrainModule' 'CNN-BiGRU' 'Conformer' 'DBConformer')

datatype=('Processed_sEEG_Vocalized_Word')

#-------------------------------------------------------------------------------------random #

for i in ${!seeds[@]}
do
    seed=${seeds[$i]}
    gpu_id=${gpus[$i]}
    echo "Running on GPU $gpu_id with seed $seed and backbone DBConformer"
    python ../train_baseline_single_modality_vm.py \
    --datatype='Processed_sEEG_Vocalized_Word' \
    --seed=$seed \
    --backbone='DBConformer' \
    --gpu_id=$gpu_id \
    --random=True \
    &
    sleep 3
done

wait
echo "subject tasks completed"

#-------------------------------------------------------------------------------------read_full #
for b in ${!backbones[@]}
do
    backbone=${backbones[$b]}
    for i in ${!seeds[@]}
    do
        seed=${seeds[$i]}
        gpu_id=${gpus[$i]}
        echo "Running on GPU $gpu_id with seed $seed and backbone $backbone"
        python ../train_baseline_single_modality_vm.py \
        --datatype='Processed_sEEG_Vocalized_Word' \
        --seed=$seed \
        --backbone=$backbone \
        --gpu_id=$gpu_id \
        &
        sleep 3
    done

    wait
    echo "subject tasks completed"
    
done

wait
echo "All tasks completed"


##########
# log->csv
##########

current_date=$(date +%Y-%m-%d)
python log2csv_acc5.py --today "$current_date" --seeds ${seeds[@]} --subjects='S01' &

wait