#!/bin/bash
###
 # @Author: WangHongbinary
 # @E-mail: hbwang22@gmail.com
 # @Date: 2025-12-10 10:43:41
 # @LastEditTime: 2025-12-23 13:34:07
 # @Description: BrainModule
### 

#####################
# Sentencehubertraw_8
#####################

seeds=(2024 2025 2026 2027 2028 2029)
gpus=(1 2 3 4 5 6)

datatype=('Processed_sEEG_Vocalized_Sentence' 'Processed_sEEG_Mimed_Sentence' 'Processed_sEEG_Imagined_Sentence' 'Processed_sEEG_Vocalized_Word')
feature_type=('Sentencehubertraw_8' 'Sentencehubert_8' 'Wordhubertraw_8' 'Wordhubert_8')

# #------------------------------------------------------------------------------------read_full random #

# for i in ${!seeds[@]}
# do
#     seed=${seeds[$i]}
#     gpu_id=${gpus[$i]}
#     echo "Running on GPU $gpu_id with seed $seed and backbone BrainModule"
#     python ../train_vm.py \
#     --datatype='Processed_sEEG_Vocalized_Word' \
#     --feature_type='Wordehubert_8' \
#     --seed=$seed \
#     --gpu_id=$gpu_id \
#     --out_channels 768 \
#     --random=True \
#     &
#         sleep 3
#     done

# wait
# echo "subject tasks completed"

#######################################################################################################
#######################################################################################################
#######################################################################################################
# read_full best
#######################################################################################################
#######################################################################################################
#######################################################################################################
#-------------------------------------------------------------------------------------read #

for i in ${!seeds[@]}
do
    seed=${seeds[$i]}
    gpu_id=${gpus[$i]}
    echo "Running on GPU $gpu_id with seed $seed and backbone BrainModule"
    python ../train_vm.py \
    --datatype='Processed_sEEG_Vocalized_Word' \
    --feature_type='Wordhubert_8' \
    --seed=$seed \
    --gpu_id=$gpu_id \
    --out_channels 768 \
    &
        sleep 3
    done

wait
echo "subject tasks completed"

#######################################################################################################
#######################################################################################################
#######################################################################################################
# log2csv
#######################################################################################################
#######################################################################################################
#######################################################################################################

current_date=$(date +%Y-%m-%d)
python log2csv_acc5.py --today "$current_date" --seeds ${seeds[@]} --subjects='S01' &

wait