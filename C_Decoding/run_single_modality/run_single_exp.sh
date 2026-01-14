#!/bin/bash
###
 # @Author: WangHongbinary
 # @E-mail: hbwang22@gmail.com
 # @Date: 2025-12-10 10:43:41
 # @LastEditTime: 2026-01-07 11:03:32
 # @Description: EEGNet | DeepConvNet | ShallowConvNet | CNN-BiGRU | Conformer | DBConformer | BrainModule
### 

##########
# hubert_8
##########

# seeds=(2024 2025 2026 2027 2028 2029)
# gpus=(1 2 3 4 5 6)

# subjects=('S01' 'S02' 'S03' 'S04' 'S05' 'S06' 'S07' 'S08' 'S09' 'S10')
# systems=('Natus/V_48' 'Natus/V_48' 'Natus/V_48' 'Natus/V_48' 'Natus/V_48' 'Natus/V_48' 'Neuracle/V_48' 'Neuracle/V_48' 'Natus/V_48' 'Neuracle/V_48')

seeds=(2024 2025 2026 2027 2028 2029)
gpus=(1 2 3 4 5 6)

subjects=('S01')
systems=('Natus/V_48')

# #-------------------------------------------------------------------------------------read_full #

# for s in ${!subjects[@]}
# do
#     subject_id=${subjects[$s]}
#     device_system=${systems[$s]}
#     for i in ${!seeds[@]}
#     do
#         seed=${seeds[$i]}
#         gpu_id=${gpus[$i]}
#         echo "Running on GPU $gpu_id with seed $seed"
#         python ../train_baseline_single_modality.py \
#         --device_system=$device_system \
#         --seed=$seed \
#         --subject_id=$subject_id \
#         --backbone='DBConformer' \
#         --gpu_id=$gpu_id \
#         --channel_path='read_full' \
#         &
#         sleep 3
#     done

#     wait
#     echo "subject tasks completed"
# done 

# wait
# echo "All tasks completed"


#-------------------------------------------------------------------------------------read_full #

for s in ${!subjects[@]}
do
    subject_id=${subjects[$s]}
    device_system=${systems[$s]}
    for i in ${!seeds[@]}
    do
        seed=${seeds[$i]}
        gpu_id=${gpus[$i]}
        echo "Running on GPU $gpu_id with seed $seed"
        python ../train_baseline_single_modality.py \
        --device_system=$device_system \
        --seed=$seed \
        --subject_id=$subject_id \
        --backbone='EEGNet' \
        --gpu_id=$gpu_id \
        --channel_path='read_full' \
        --forward_feature=True \
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
python log2csv.py --today "$current_date" --seeds ${seeds[@]} --subjects ${subjects[@]} &

wait