#!/bin/bash
###
 # @Author: WangHongbinary
 # @E-mail: hbwang22@gmail.com
 # @Date: 2025-12-10 10:43:41
 # @LastEditTime: 2025-12-30 19:55:59
 # @Description: EEGNet | DeepConvNet | ShallowConvNet | CNN-BiGRU | Conformer | DBConformer
### 

#####################
# Sentencehubertraw_8
#####################

seeds=(2024 2025 2026 2027 2028 2029)
gpus=(5 6 5 6 5 6)
backbones=('EEGNet' 'DeepConvNet' 'ShallowConvNet' 'CNN-BiGRU' 'Conformer' 'DBConformer')
datatype=('Processed_sEEG_Vocalized_Sentence' 'Processed_sEEG_Mimed_Sentence' 'Processed_sEEG_Imagined_Sentence' 'Processed_sEEG_Vocalized_Word')
feature_type=('Sentencehubertraw_8' 'Sentencehubert_8' 'Wordhubertraw_8' 'Wordhubert_8')

# #-------------------------------------------------------------------------------------random #

# for i in ${!seeds[@]}
# do
#     seed=${seeds[$i]}
#     gpu_id=${gpus[$i]}
#     echo "Running on GPU $gpu_id with seed $seed and backbone CNN-BiGRU"
#     python ../train_baseline_multi_modality_vm.py \
#     --datatype='Processed_sEEG_Vocalized_Word' \
#     --feature_type='Wordhubertraw_8' \
#     --seed=$seed \
#     --backbone='CNN-BiGRU' \
#     --gpu_id=$gpu_id \
#     --out_dim 256 \
#     --out_time 64 \
#     --random=True \
#     &
#     sleep 3
# done

# wait
# echo "All tasks completed"

#######################################################################################################
#######################################################################################################
#######################################################################################################
# read_full best
#######################################################################################################
#######################################################################################################
#######################################################################################################
# #-------------------------------------------------------------------------------------read #
# for b in ${!backbones[@]}
# do
#     backbone=${backbones[$b]}
#     for i in ${!seeds[@]}
#     do
#         seed=${seeds[$i]}
#         gpu_id=${gpus[$i]}
#         echo "Running on GPU $gpu_id with seed $seed and backbone $backbone"
#         python ../train_baseline_multi_modality_vm.py \
#         --datatype='Processed_sEEG_Vocalized_Word' \
#         --feature_type='Wordhubertraw_8' \
#         --seed=$seed \
#         --backbone=$backbone \
#         --gpu_id=$gpu_id \
#         --out_dim 256 \
#         --out_time 64 \
#         &
#         sleep 3
#     done

#     wait
#     echo "subject tasks completed"
# done 

# wait
# echo "All tasks completed"


#######################################################################################################
#######################################################################################################
#######################################################################################################
# temperature 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1
#######################################################################################################
#######################################################################################################
#######################################################################################################
# #-------------------------------------------------------------------------------------0.005 #

# for i in ${!seeds[@]}
# do
#     seed=${seeds[$i]}
#     gpu_id=${gpus[$i]}
#     echo "Running on GPU $gpu_id with seed $seed and backbone CNN-BiGRU"
#     python ../train_baseline_multi_modality_vm.py \
#     --datatype='Processed_sEEG_Vocalized_Word' \
#     --feature_type='Wordhubertraw_8' \
#     --seed=$seed \
#     --backbone='CNN-BiGRU' \
#     --gpu_id=$gpu_id \
#     --out_dim 256 \
#     --out_time 64 \
#     --temperature 200 \
#     &
#     sleep 3
# done

# wait
# echo "All tasks completed"

# #-------------------------------------------------------------------------------------0.01 #

# for i in ${!seeds[@]}
# do
#     seed=${seeds[$i]}
#     gpu_id=${gpus[$i]}
#     echo "Running on GPU $gpu_id with seed $seed and backbone CNN-BiGRU"
#     python ../train_baseline_multi_modality_vm.py \
#     --datatype='Processed_sEEG_Vocalized_Word' \
#     --feature_type='Wordhubertraw_8' \
#     --seed=$seed \
#     --backbone='CNN-BiGRU' \
#     --gpu_id=$gpu_id \
#     --out_dim 256 \
#     --out_time 64 \
#     --temperature 100 \
#     &
#     sleep 3
# done

# wait
# echo "All tasks completed"

# #-------------------------------------------------------------------------------------0.02 #

# for i in ${!seeds[@]}
# do
#     seed=${seeds[$i]}
#     gpu_id=${gpus[$i]}
#     echo "Running on GPU $gpu_id with seed $seed and backbone CNN-BiGRU"
#     python ../train_baseline_multi_modality_vm.py \
#     --datatype='Processed_sEEG_Vocalized_Word' \
#     --feature_type='Wordhubertraw_8' \
#     --seed=$seed \
#     --backbone='CNN-BiGRU' \
#     --gpu_id=$gpu_id \
#     --out_dim 256 \
#     --out_time 64 \
#     --temperature 50 \
#     &
#     sleep 3
# done

# wait
# echo "All tasks completed"

# #-------------------------------------------------------------------------------------0.05 #

# for i in ${!seeds[@]}
# do
#     seed=${seeds[$i]}
#     gpu_id=${gpus[$i]}
#     echo "Running on GPU $gpu_id with seed $seed and backbone CNN-BiGRU"
#     python ../train_baseline_multi_modality_vm.py \
#     --datatype='Processed_sEEG_Vocalized_Word' \
#     --feature_type='Wordhubertraw_8' \
#     --seed=$seed \
#     --backbone='CNN-BiGRU' \
#     --gpu_id=$gpu_id \
#     --out_dim 256 \
#     --out_time 64 \
#     --temperature 20 \
#     &
#     sleep 3
# done

# wait
# echo "All tasks completed"

# #-------------------------------------------------------------------------------------0.1 #

# for i in ${!seeds[@]}
# do
#     seed=${seeds[$i]}
#     gpu_id=${gpus[$i]}
#     echo "Running on GPU $gpu_id with seed $seed and backbone CNN-BiGRU"
#     python ../train_baseline_multi_modality_vm.py \
#     --datatype='Processed_sEEG_Vocalized_Word' \
#     --feature_type='Wordhubertraw_8' \
#     --seed=$seed \
#     --backbone='CNN-BiGRU' \
#     --gpu_id=$gpu_id \
#     --out_dim 256 \
#     --out_time 64 \
#     --temperature 10 \
#     &
#     sleep 3
# done

# wait
# echo "All tasks completed"

# #-------------------------------------------------------------------------------------0.2 #

# for i in ${!seeds[@]}
# do
#     seed=${seeds[$i]}
#     gpu_id=${gpus[$i]}
#     echo "Running on GPU $gpu_id with seed $seed and backbone CNN-BiGRU"
#     python ../train_baseline_multi_modality_vm.py \
#     --datatype='Processed_sEEG_Vocalized_Word' \
#     --feature_type='Wordhubertraw_8' \
#     --seed=$seed \
#     --backbone='CNN-BiGRU' \
#     --gpu_id=$gpu_id \
#     --out_dim 256 \
#     --out_time 64 \
#     --temperature 5 \
#     &
#     sleep 3
# done

# wait
# echo "All tasks completed"

# #-------------------------------------------------------------------------------------0.5 #

# for i in ${!seeds[@]}
# do
#     seed=${seeds[$i]}
#     gpu_id=${gpus[$i]}
#     echo "Running on GPU $gpu_id with seed $seed and backbone CNN-BiGRU"
#     python ../train_baseline_multi_modality_vm.py \
#     --datatype='Processed_sEEG_Vocalized_Word' \
#     --feature_type='Wordhubertraw_8' \
#     --seed=$seed \
#     --backbone='CNN-BiGRU' \
#     --gpu_id=$gpu_id \
#     --out_dim 256 \
#     --out_time 64 \
#     --temperature 2 \
#     &
#     sleep 3
# done

# wait
# echo "All tasks completed"

# #-------------------------------------------------------------------------------------1 #

# for i in ${!seeds[@]}
# do
#     seed=${seeds[$i]}
#     gpu_id=${gpus[$i]}
#     echo "Running on GPU $gpu_id with seed $seed and backbone CNN-BiGRU"
#     python ../train_baseline_multi_modality_vm.py \
#     --datatype='Processed_sEEG_Vocalized_Word' \
#     --feature_type='Wordhubertraw_8' \
#     --seed=$seed \
#     --backbone='CNN-BiGRU' \
#     --gpu_id=$gpu_id \
#     --out_dim 256 \
#     --out_time 64 \
#     --temperature 1 \
#     &
#     sleep 3
# done

# wait
# echo "All tasks completed"

#######################################################################################################
#######################################################################################################
#######################################################################################################
# temperature 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1
#######################################################################################################
#######################################################################################################
#######################################################################################################
#-------------------------------------------------------------------------------------0.005 #

for i in ${!seeds[@]}
do
    seed=${seeds[$i]}
    gpu_id=${gpus[$i]}
    echo "Running on GPU $gpu_id with seed $seed and backbone EEGNet"
    python ../train_baseline_multi_modality_vm.py \
    --datatype='Processed_sEEG_Vocalized_Sentence' \
    --feature_type='Sentencehubertraw_8' \
    --seed=$seed \
    --backbone='EEGNet' \
    --gpu_id=$gpu_id \
    --out_dim 256 \
    --out_time 64 \
    --temperature 200 \
    &
    sleep 3
done

wait
echo "All tasks completed"

#-------------------------------------------------------------------------------------0.01 #

for i in ${!seeds[@]}
do
    seed=${seeds[$i]}
    gpu_id=${gpus[$i]}
    echo "Running on GPU $gpu_id with seed $seed and backbone EEGNet"
    python ../train_baseline_multi_modality_vm.py \
    --datatype='Processed_sEEG_Vocalized_Sentence' \
    --feature_type='Sentencehubertraw_8' \
    --seed=$seed \
    --backbone='EEGNet' \
    --gpu_id=$gpu_id \
    --out_dim 256 \
    --out_time 64 \
    --temperature 100 \
    &
    sleep 3
done

wait
echo "All tasks completed"

#-------------------------------------------------------------------------------------0.02 #

for i in ${!seeds[@]}
do
    seed=${seeds[$i]}
    gpu_id=${gpus[$i]}
    echo "Running on GPU $gpu_id with seed $seed and backbone EEGNet"
    python ../train_baseline_multi_modality_vm.py \
    --datatype='Processed_sEEG_Vocalized_Sentence' \
    --feature_type='Sentencehubertraw_8' \
    --seed=$seed \
    --backbone='EEGNet' \
    --gpu_id=$gpu_id \
    --out_dim 256 \
    --out_time 64 \
    --temperature 50 \
    &
    sleep 3
done

wait
echo "All tasks completed"

#-------------------------------------------------------------------------------------0.05 #

for i in ${!seeds[@]}
do
    seed=${seeds[$i]}
    gpu_id=${gpus[$i]}
    echo "Running on GPU $gpu_id with seed $seed and backbone EEGNet"
    python ../train_baseline_multi_modality_vm.py \
    --datatype='Processed_sEEG_Vocalized_Sentence' \
    --feature_type='Sentencehubertraw_8' \
    --seed=$seed \
    --backbone='EEGNet' \
    --gpu_id=$gpu_id \
    --out_dim 256 \
    --out_time 64 \
    --temperature 20 \
    &
    sleep 3
done

wait
echo "All tasks completed"

#-------------------------------------------------------------------------------------0.1 #

for i in ${!seeds[@]}
do
    seed=${seeds[$i]}
    gpu_id=${gpus[$i]}
    echo "Running on GPU $gpu_id with seed $seed and backbone EEGNet"
    python ../train_baseline_multi_modality_vm.py \
    --datatype='Processed_sEEG_Vocalized_Sentence' \
    --feature_type='Sentencehubertraw_8' \
    --seed=$seed \
    --backbone='EEGNet' \
    --gpu_id=$gpu_id \
    --out_dim 256 \
    --out_time 64 \
    --temperature 10 \
    &
    sleep 3
done

wait
echo "All tasks completed"

#-------------------------------------------------------------------------------------0.2 #

for i in ${!seeds[@]}
do
    seed=${seeds[$i]}
    gpu_id=${gpus[$i]}
    echo "Running on GPU $gpu_id with seed $seed and backbone EEGNet"
    python ../train_baseline_multi_modality_vm.py \
    --datatype='Processed_sEEG_Vocalized_Sentence' \
    --feature_type='Sentencehubertraw_8' \
    --seed=$seed \
    --backbone='EEGNet' \
    --gpu_id=$gpu_id \
    --out_dim 256 \
    --out_time 64 \
    --temperature 5 \
    &
    sleep 3
done

wait
echo "All tasks completed"

#-------------------------------------------------------------------------------------0.5 #

for i in ${!seeds[@]}
do
    seed=${seeds[$i]}
    gpu_id=${gpus[$i]}
    echo "Running on GPU $gpu_id with seed $seed and backbone EEGNet"
    python ../train_baseline_multi_modality_vm.py \
    --datatype='Processed_sEEG_Vocalized_Sentence' \
    --feature_type='Sentencehubertraw_8' \
    --seed=$seed \
    --backbone='EEGNet' \
    --gpu_id=$gpu_id \
    --out_dim 256 \
    --out_time 64 \
    --temperature 2 \
    &
    sleep 3
done

wait
echo "All tasks completed"

#-------------------------------------------------------------------------------------1 #

for i in ${!seeds[@]}
do
    seed=${seeds[$i]}
    gpu_id=${gpus[$i]}
    echo "Running on GPU $gpu_id with seed $seed and backbone EEGNet"
    python ../train_baseline_multi_modality_vm.py \
    --datatype='Processed_sEEG_Vocalized_Sentence' \
    --feature_type='Sentencehubertraw_8' \
    --seed=$seed \
    --backbone='EEGNet' \
    --gpu_id=$gpu_id \
    --out_dim 256 \
    --out_time 64 \
    --temperature 1 \
    &
    sleep 3
done

wait
echo "All tasks completed"

#######################################################################################################
#######################################################################################################
#######################################################################################################
# temperature 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1
#######################################################################################################
#######################################################################################################
#######################################################################################################

# #-------------------------------------------------------------------------------------Wordhubertraw_0 #

# for i in ${!seeds[@]}
# do
#     seed=${seeds[$i]}
#     gpu_id=${gpus[$i]}
#     echo "Running on GPU $gpu_id with seed $seed and backbone CNN-BiGRU"
#     python ../train_baseline_multi_modality_vm.py \
#     --datatype='Processed_sEEG_Vocalized_Word' \
#     --feature_type='Wordhubertraw_0' \
#     --seed=$seed \
#     --backbone='CNN-BiGRU' \
#     --gpu_id=$gpu_id \
#     --out_dim 256 \
#     --out_time 64 \
#     &
#     sleep 3
# done

# wait
# echo "All tasks completed"

# #-------------------------------------------------------------------------------------Wordhubertraw_2 #

# for i in ${!seeds[@]}
# do
#     seed=${seeds[$i]}
#     gpu_id=${gpus[$i]}
#     echo "Running on GPU $gpu_id with seed $seed and backbone CNN-BiGRU"
#     python ../train_baseline_multi_modality_vm.py \
#     --datatype='Processed_sEEG_Vocalized_Word' \
#     --feature_type='Wordhubertraw_2' \
#     --seed=$seed \
#     --backbone='CNN-BiGRU' \
#     --gpu_id=$gpu_id \
#     --out_dim 256 \
#     --out_time 64 \
#     &
#     sleep 3
# done

# wait
# echo "All tasks completed"

# #-------------------------------------------------------------------------------------Wordhubertraw_4 #

# for i in ${!seeds[@]}
# do
#     seed=${seeds[$i]}
#     gpu_id=${gpus[$i]}
#     echo "Running on GPU $gpu_id with seed $seed and backbone CNN-BiGRU"
#     python ../train_baseline_multi_modality_vm.py \
#     --datatype='Processed_sEEG_Vocalized_Word' \
#     --feature_type='Wordhubertraw_4' \
#     --seed=$seed \
#     --backbone='CNN-BiGRU' \
#     --gpu_id=$gpu_id \
#     --out_dim 256 \
#     --out_time 64 \
#     &
#     sleep 3
# done

# wait
# echo "All tasks completed"

# #-------------------------------------------------------------------------------------Wordhubertraw_6 #

# for i in ${!seeds[@]}
# do
#     seed=${seeds[$i]}
#     gpu_id=${gpus[$i]}
#     echo "Running on GPU $gpu_id with seed $seed and backbone CNN-BiGRU"
#     python ../train_baseline_multi_modality_vm.py \
#     --datatype='Processed_sEEG_Vocalized_Word' \
#     --feature_type='Wordhubertraw_6' \
#     --seed=$seed \
#     --backbone='CNN-BiGRU' \
#     --gpu_id=$gpu_id \
#     --out_dim 256 \
#     --out_time 64 \
#     &
#     sleep 3
# done

# wait
# echo "All tasks completed"

# #-------------------------------------------------------------------------------------Wordhubertraw_8 #

# for i in ${!seeds[@]}
# do
#     seed=${seeds[$i]}
#     gpu_id=${gpus[$i]}
#     echo "Running on GPU $gpu_id with seed $seed and backbone CNN-BiGRU"
#     python ../train_baseline_multi_modality_vm.py \
#     --datatype='Processed_sEEG_Vocalized_Word' \
#     --feature_type='Wordhubertraw_8' \
#     --seed=$seed \
#     --backbone='CNN-BiGRU' \
#     --gpu_id=$gpu_id \
#     --out_dim 256 \
#     --out_time 64 \
#     &
#     sleep 3
# done

# wait
# echo "All tasks completed"

# #-------------------------------------------------------------------------------------Wordhubertraw_10 #

# for i in ${!seeds[@]}
# do
#     seed=${seeds[$i]}
#     gpu_id=${gpus[$i]}
#     echo "Running on GPU $gpu_id with seed $seed and backbone CNN-BiGRU"
#     python ../train_baseline_multi_modality_vm.py \
#     --datatype='Processed_sEEG_Vocalized_Word' \
#     --feature_type='Wordhubertraw_10' \
#     --seed=$seed \
#     --backbone='CNN-BiGRU' \
#     --gpu_id=$gpu_id \
#     --out_dim 256 \
#     --out_time 64 \
#     &
#     sleep 3
# done

# wait
# echo "All tasks completed"

#######################################################################################################
#######################################################################################################
#######################################################################################################
# log2csv
#######################################################################################################
#######################################################################################################
#######################################################################################################

current_date=$(date +%Y-%m-%d)
python log2csv_acc3.py --today "$current_date" --seeds ${seeds[@]} --subjects='S01' &

wait