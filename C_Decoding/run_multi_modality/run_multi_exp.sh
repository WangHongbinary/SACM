#!/bin/bash
###
 # @Author: WangHongbinary
 # @E-mail: hbwang22@gmail.com
 # @Date: 2025-12-10 10:43:41
 # @LastEditTime: 2026-01-07 15:38:41
 # @Description: EEGNet | DeepConvNet | ShallowConvNet | CNN-BiGRU | Conformer | DBConformer
### 

##########
# hubert_8
##########

seeds=(2024 2025 2026 2027 2028 2029)
gpus=(1 2 3 4 5 6)

# subjects=('S01' 'S02' 'S03' 'S04' 'S05' 'S06' 'S07' 'S08' 'S09' 'S10')
# systems=('Natus/V_48' 'Natus/V_48' 'Natus/V_48' 'Natus/V_48' 'Natus/V_48' 'Natus/V_48' 'Neuracle/V_48' 'Neuracle/V_48' 'Natus/V_48' 'Neuracle/V_48')

subjects=('S01')
systems=('Natus/V_48')

#######################################################################################################
#######################################################################################################
#######################################################################################################
# read_full random
#######################################################################################################
#######################################################################################################
#######################################################################################################

# #-------------------------------------------------------------------------------------read_full_random #

# for s in ${!subjects[@]}
# do
#     subject_id=${subjects[$s]}
#     device_system=${systems[$s]}
#     for i in ${!seeds[@]}
#     do
#         seed=${seeds[$i]}
#         gpu_id=${gpus[$i]}
#         echo "Running on GPU $gpu_id with seed $seed"
#         python ../train_baseline_multi_modality.py \
#         --device_system=$device_system \
#         --seed=$seed \
#         --subject_id=$subject_id \
#         --backbone='DBConformer' \
#         --gpu_id=$gpu_id \
#         --channel_path='read_full' \
#         --out_dim 256 \
#         --out_time 32 \
#         --random=True \
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
# read_full best
#######################################################################################################
#######################################################################################################
#######################################################################################################
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
        python ../train_baseline_multi_modality.py \
        --device_system=$device_system \
        --seed=$seed \
        --subject_id=$subject_id \
        --backbone='EEGNet' \
        --gpu_id=$gpu_id \
        --channel_path='read_full' \
        --out_dim 256 \
        --out_time 32 \
        --forward_feature=True \
        &
        sleep 3
    done

    wait
    echo "subject tasks completed"
done

wait
echo "All tasks completed"

#######################################################################################################
#######################################################################################################
#######################################################################################################
# single channel CAR & Bipolar
#######################################################################################################
#######################################################################################################
#######################################################################################################

# #-------------------------------------------------------------------------------------read_single_A #

# for s in ${!subjects[@]}
# do
#     subject_id=${subjects[$s]}
#     device_system=${systems[$s]}
#     for i in ${!seeds[@]}
#     do
#         seed=${seeds[$i]}
#         gpu_id=${gpus[$i]}
#         echo "Running on GPU $gpu_id with seed $seed"
#         python ../train_baseline_multi_modality.py \
#         --device_system=$device_system \
#         --seed=$seed \
#         --subject_id=$subject_id \
#         --backbone='DBConformer' \
#         --gpu_id=$gpu_id \
#         --channel_path='read_single_A' \
#         --out_dim 256 \
#         --out_time 32 \
#         &
#         sleep 3
#     done

#     wait
#     echo "subject tasks completed"
# done

# wait
# echo "All tasks completed"

# #-------------------------------------------------------------------------------------read_single_B #

# for s in ${!subjects[@]}
# do
#     subject_id=${subjects[$s]}
#     device_system=${systems[$s]}
#     for i in ${!seeds[@]}
#     do
#         seed=${seeds[$i]}
#         gpu_id=${gpus[$i]}
#         echo "Running on GPU $gpu_id with seed $seed"
#         python ../train_baseline_multi_modality.py \
#         --device_system=$device_system \
#         --seed=$seed \
#         --subject_id=$subject_id \
#         --backbone='DBConformer' \
#         --gpu_id=$gpu_id \
#         --channel_path='read_single_B' \
#         --out_dim 256 \
#         --out_time 32 \
#         &
#         sleep 3
#     done

#     wait
#     echo "subject tasks completed"
# done

# wait
# echo "All tasks completed"

# #-------------------------------------------------------------------------------------read_single_C #

# for s in ${!subjects[@]}
# do
#     subject_id=${subjects[$s]}
#     device_system=${systems[$s]}
#     for i in ${!seeds[@]}
#     do
#         seed=${seeds[$i]}
#         gpu_id=${gpus[$i]}
#         echo "Running on GPU $gpu_id with seed $seed"
#         python ../train_baseline_multi_modality.py \
#         --device_system=$device_system \
#         --seed=$seed \
#         --subject_id=$subject_id \
#         --backbone='DBConformer' \
#         --gpu_id=$gpu_id \
#         --channel_path='read_single_C' \
#         --out_dim 256 \
#         --out_time 32 \
#         &
#         sleep 3
#     done

#     wait
#     echo "subject tasks completed"
# done

# wait
# echo "All tasks completed"

# #-------------------------------------------------------------------------------------read_single_D #

# for s in ${!subjects[@]}
# do
#     subject_id=${subjects[$s]}
#     device_system=${systems[$s]}
#     for i in ${!seeds[@]}
#     do
#         seed=${seeds[$i]}
#         gpu_id=${gpus[$i]}
#         echo "Running on GPU $gpu_id with seed $seed"
#         python ../train_baseline_multi_modality.py \
#         --device_system=$device_system \
#         --seed=$seed \
#         --subject_id=$subject_id \
#         --backbone='DBConformer' \
#         --gpu_id=$gpu_id \
#         --channel_path='read_single_D' \
#         --out_dim 256 \
#         --out_time 32 \
#         &
#         sleep 3
#     done

#     wait
#     echo "subject tasks completed"
# done

# wait
# echo "All tasks completed"

# #-------------------------------------------------------------------------------------read_single_E #

# for s in ${!subjects[@]}
# do
#     subject_id=${subjects[$s]}
#     device_system=${systems[$s]}
#     for i in ${!seeds[@]}
#     do
#         seed=${seeds[$i]}
#         gpu_id=${gpus[$i]}
#         echo "Running on GPU $gpu_id with seed $seed"
#         python ../train_baseline_multi_modality.py \
#         --device_system=$device_system \
#         --seed=$seed \
#         --subject_id=$subject_id \
#         --backbone='DBConformer' \
#         --gpu_id=$gpu_id \
#         --channel_path='read_single_E' \
#         --out_dim 256 \
#         --out_time 32 \
#         &
#         sleep 3
#     done

#     wait
#     echo "subject tasks completed"
# done

# wait
# echo "All tasks completed"

# #-------------------------------------------------------------------------------------read_single_F #

# for s in ${!subjects[@]}
# do
#     subject_id=${subjects[$s]}
#     device_system=${systems[$s]}
#     for i in ${!seeds[@]}
#     do
#         seed=${seeds[$i]}
#         gpu_id=${gpus[$i]}
#         echo "Running on GPU $gpu_id with seed $seed"
#         python ../train_baseline_multi_modality.py \
#         --device_system=$device_system \
#         --seed=$seed \
#         --subject_id=$subject_id \
#         --backbone='DBConformer' \
#         --gpu_id=$gpu_id \
#         --channel_path='read_single_F' \
#         --out_dim 256 \
#         --out_time 32 \
#         &
#         sleep 3
#     done

#     wait
#     echo "subject tasks completed"
# done

# wait
# echo "All tasks completed"

# #-------------------------------------------------------------------------------------read_single_G #

# for s in ${!subjects[@]}
# do
#     subject_id=${subjects[$s]}
#     device_system=${systems[$s]}
#     for i in ${!seeds[@]}
#     do
#         seed=${seeds[$i]}
#         gpu_id=${gpus[$i]}
#         echo "Running on GPU $gpu_id with seed $seed"
#         python ../train_baseline_multi_modality.py \
#         --device_system=$device_system \
#         --seed=$seed \
#         --subject_id=$subject_id \
#         --backbone='DBConformer' \
#         --gpu_id=$gpu_id \
#         --channel_path='read_single_G' \
#         --out_dim 256 \
#         --out_time 32 \
#         &
#         sleep 3
#     done

#     wait
#     echo "subject tasks completed"
# done

# wait
# echo "All tasks completed"


# #-------------------------------------------------------------------------------------read_full_bipolar_random #

# for s in ${!subjects[@]}
# do
#     subject_id=${subjects[$s]}
#     device_system=${systems[$s]}
#     for i in ${!seeds[@]}
#     do
#         seed=${seeds[$i]}
#         gpu_id=${gpus[$i]}
#         echo "Running on GPU $gpu_id with seed $seed"
#         python ../train_baseline_multi_modality.py \
#         --device_system=$device_system \
#         --seed=$seed \
#         --subject_id=$subject_id \
#         --backbone='DBConformer' \
#         --gpu_id=$gpu_id \
#         --channel_path='read_full_bipolar' \
#         --out_dim 256 \
#         --out_time 32 \
#         --random=True \
#         &
#         sleep 3
#     done

#     wait
#     echo "subject tasks completed"
# done

# wait
# echo "All tasks completed"

# #-------------------------------------------------------------------------------------read_full_bipolar #

# for s in ${!subjects[@]}
# do
#     subject_id=${subjects[$s]}
#     device_system=${systems[$s]}
#     for i in ${!seeds[@]}
#     do
#         seed=${seeds[$i]}
#         gpu_id=${gpus[$i]}
#         echo "Running on GPU $gpu_id with seed $seed"
#         python ../train_baseline_multi_modality.py \
#         --device_system=$device_system \
#         --seed=$seed \
#         --subject_id=$subject_id \
#         --backbone='DBConformer' \
#         --gpu_id=$gpu_id \
#         --channel_path='read_full_bipolar' \
#         --out_dim 256 \
#         --out_time 32 \
#         &
#         sleep 3
#     done

#     wait
#     echo "subject tasks completed"
# done

# wait
# echo "All tasks completed"

# #-------------------------------------------------------------------------------------read_single_A_bipolar #

# for s in ${!subjects[@]}
# do
#     subject_id=${subjects[$s]}
#     device_system=${systems[$s]}
#     for i in ${!seeds[@]}
#     do
#         seed=${seeds[$i]}
#         gpu_id=${gpus[$i]}
#         echo "Running on GPU $gpu_id with seed $seed"
#         python ../train_baseline_multi_modality.py \
#         --device_system=$device_system \
#         --seed=$seed \
#         --subject_id=$subject_id \
#         --backbone='DBConformer' \
#         --gpu_id=$gpu_id \
#         --channel_path='read_single_A_bipolar' \
#         --out_dim 256 \
#         --out_time 32 \
#         &
#         sleep 3
#     done

#     wait
#     echo "subject tasks completed"
# done

# wait
# echo "All tasks completed"

# #-------------------------------------------------------------------------------------read_single_B_bipolar #

# for s in ${!subjects[@]}
# do
#     subject_id=${subjects[$s]}
#     device_system=${systems[$s]}
#     for i in ${!seeds[@]}
#     do
#         seed=${seeds[$i]}
#         gpu_id=${gpus[$i]}
#         echo "Running on GPU $gpu_id with seed $seed"
#         python ../train_baseline_multi_modality.py \
#         --device_system=$device_system \
#         --seed=$seed \
#         --subject_id=$subject_id \
#         --backbone='DBConformer' \
#         --gpu_id=$gpu_id \
#         --channel_path='read_single_B_bipolar' \
#         --out_dim 256 \
#         --out_time 32 \
#         &
#         sleep 3
#     done

#     wait
#     echo "subject tasks completed"
# done

# wait
# echo "All tasks completed"

# #-------------------------------------------------------------------------------------read_single_C_bipolar #

# for s in ${!subjects[@]}
# do
#     subject_id=${subjects[$s]}
#     device_system=${systems[$s]}
#     for i in ${!seeds[@]}
#     do
#         seed=${seeds[$i]}
#         gpu_id=${gpus[$i]}
#         echo "Running on GPU $gpu_id with seed $seed"
#         python ../train_baseline_multi_modality.py \
#         --device_system=$device_system \
#         --seed=$seed \
#         --subject_id=$subject_id \
#         --backbone='DBConformer' \
#         --gpu_id=$gpu_id \
#         --channel_path='read_single_C_bipolar' \
#         --out_dim 256 \
#         --out_time 32 \
#         &
#         sleep 3
#     done

#     wait
#     echo "subject tasks completed"
# done

# wait
# echo "All tasks completed"

# #-------------------------------------------------------------------------------------read_single_D_bipolar #

# for s in ${!subjects[@]}
# do
#     subject_id=${subjects[$s]}
#     device_system=${systems[$s]}
#     for i in ${!seeds[@]}
#     do
#         seed=${seeds[$i]}
#         gpu_id=${gpus[$i]}
#         echo "Running on GPU $gpu_id with seed $seed"
#         python ../train_baseline_multi_modality.py \
#         --device_system=$device_system \
#         --seed=$seed \
#         --subject_id=$subject_id \
#         --backbone='DBConformer' \
#         --gpu_id=$gpu_id \
#         --channel_path='read_single_D_bipolar' \
#         --out_dim 256 \
#         --out_time 32 \
#         &
#         sleep 3
#     done

#     wait
#     echo "subject tasks completed"
# done

# wait
# echo "All tasks completed"

# #-------------------------------------------------------------------------------------read_single_E_bipolar #

# for s in ${!subjects[@]}
# do
#     subject_id=${subjects[$s]}
#     device_system=${systems[$s]}
#     for i in ${!seeds[@]}
#     do
#         seed=${seeds[$i]}
#         gpu_id=${gpus[$i]}
#         echo "Running on GPU $gpu_id with seed $seed"
#         python ../train_baseline_multi_modality.py \
#         --device_system=$device_system \
#         --seed=$seed \
#         --subject_id=$subject_id \
#         --backbone='DBConformer' \
#         --gpu_id=$gpu_id \
#         --channel_path='read_single_E_bipolar' \
#         --out_dim 256 \
#         --out_time 32 \
#         &
#         sleep 3
#     done

#     wait
#     echo "subject tasks completed"
# done

# wait
# echo "All tasks completed"

# #-------------------------------------------------------------------------------------read_single_F_bipolar #

# for s in ${!subjects[@]}
# do
#     subject_id=${subjects[$s]}
#     device_system=${systems[$s]}
#     for i in ${!seeds[@]}
#     do
#         seed=${seeds[$i]}
#         gpu_id=${gpus[$i]}
#         echo "Running on GPU $gpu_id with seed $seed"
#         python ../train_baseline_multi_modality.py \
#         --device_system=$device_system \
#         --seed=$seed \
#         --subject_id=$subject_id \
#         --backbone='DBConformer' \
#         --gpu_id=$gpu_id \
#         --channel_path='read_single_F_bipolar' \
#         --out_dim 256 \
#         --out_time 32 \
#         &
#         sleep 3
#     done

#     wait
#     echo "subject tasks completed"
# done

# wait
# echo "All tasks completed"

# #-------------------------------------------------------------------------------------read_single_G_bipolar #

# for s in ${!subjects[@]}
# do
#     subject_id=${subjects[$s]}
#     device_system=${systems[$s]}
#     for i in ${!seeds[@]}
#     do
#         seed=${seeds[$i]}
#         gpu_id=${gpus[$i]}
#         echo "Running on GPU $gpu_id with seed $seed"
#         python ../train_baseline_multi_modality.py \
#         --device_system=$device_system \
#         --seed=$seed \
#         --subject_id=$subject_id \
#         --backbone='DBConformer' \
#         --gpu_id=$gpu_id \
#         --channel_path='read_single_G_bipolar' \
#         --out_dim 256 \
#         --out_time 32 \
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
# temperature 0.005 0.01 0.02 0.1 0.2 0.5 1
#######################################################################################################
#######################################################################################################
#######################################################################################################

# #-------------------------------------------------------------------------------------read_full 0.005 #

# for s in ${!subjects[@]}
# do
#     subject_id=${subjects[$s]}
#     device_system=${systems[$s]}
#     for i in ${!seeds[@]}
#     do
#         seed=${seeds[$i]}
#         gpu_id=${gpus[$i]}
#         echo "Running on GPU $gpu_id with seed $seed"
#         python ../train_baseline_multi_modality.py \
#         --device_system=$device_system \
#         --seed=$seed \
#         --subject_id=$subject_id \
#         --backbone='DBConformer' \
#         --gpu_id=$gpu_id \
#         --channel_path='read_full' \
#         --out_dim 256 \
#         --out_time 32 \
#         --temperature 200 \
#         &
#         sleep 3
#     done

#     wait
#     echo "subject tasks completed"
# done

# wait
# echo "All tasks completed"

# #-------------------------------------------------------------------------------------read_full 0.01 #

# for s in ${!subjects[@]}
# do
#     subject_id=${subjects[$s]}
#     device_system=${systems[$s]}
#     for i in ${!seeds[@]}
#     do
#         seed=${seeds[$i]}
#         gpu_id=${gpus[$i]}
#         echo "Running on GPU $gpu_id with seed $seed"
#         python ../train_baseline_multi_modality.py \
#         --device_system=$device_system \
#         --seed=$seed \
#         --subject_id=$subject_id \
#         --backbone='DBConformer' \
#         --gpu_id=$gpu_id \
#         --channel_path='read_full' \
#         --out_dim 256 \
#         --out_time 32 \
#         --temperature 100 \
#         &
#         sleep 3
#     done

#     wait
#     echo "subject tasks completed"
# done

# wait
# echo "All tasks completed"


# #-------------------------------------------------------------------------------------read_full 0.02 #

# for s in ${!subjects[@]}
# do
#     subject_id=${subjects[$s]}
#     device_system=${systems[$s]}
#     for i in ${!seeds[@]}
#     do
#         seed=${seeds[$i]}
#         gpu_id=${gpus[$i]}
#         echo "Running on GPU $gpu_id with seed $seed"
#         python ../train_baseline_multi_modality.py \
#         --device_system=$device_system \
#         --seed=$seed \
#         --subject_id=$subject_id \
#         --backbone='DBConformer' \
#         --gpu_id=$gpu_id \
#         --channel_path='read_full' \
#         --out_dim 256 \
#         --out_time 32 \
#         --temperature 50 \
#         &
#         sleep 3
#     done

#     wait
#     echo "subject tasks completed"
# done

# wait
# echo "All tasks completed"


# #-------------------------------------------------------------------------------------read_full 0.1 #

# for s in ${!subjects[@]}
# do
#     subject_id=${subjects[$s]}
#     device_system=${systems[$s]}
#     for i in ${!seeds[@]}
#     do
#         seed=${seeds[$i]}
#         gpu_id=${gpus[$i]}
#         echo "Running on GPU $gpu_id with seed $seed"
#         python ../train_baseline_multi_modality.py \
#         --device_system=$device_system \
#         --seed=$seed \
#         --subject_id=$subject_id \
#         --backbone='DBConformer' \
#         --gpu_id=$gpu_id \
#         --channel_path='read_full' \
#         --out_dim 256 \
#         --out_time 32 \
#         --temperature 10 \
#         &
#         sleep 3
#     done

#     wait
#     echo "subject tasks completed"
# done

# wait
# echo "All tasks completed"


# #-------------------------------------------------------------------------------------read_full 0.2#

# for s in ${!subjects[@]}
# do
#     subject_id=${subjects[$s]}
#     device_system=${systems[$s]}
#     for i in ${!seeds[@]}
#     do
#         seed=${seeds[$i]}
#         gpu_id=${gpus[$i]}
#         echo "Running on GPU $gpu_id with seed $seed"
#         python ../train_baseline_multi_modality.py \
#         --device_system=$device_system \
#         --seed=$seed \
#         --subject_id=$subject_id \
#         --backbone='DBConformer' \
#         --gpu_id=$gpu_id \
#         --channel_path='read_full' \
#         --out_dim 256 \
#         --out_time 32 \
#         --temperature 5 \
#         &
#         sleep 3
#     done

#     wait
#     echo "subject tasks completed"
# done

# wait
# echo "All tasks completed"


# #-------------------------------------------------------------------------------------read_full 0.5#

# for s in ${!subjects[@]}
# do
#     subject_id=${subjects[$s]}
#     device_system=${systems[$s]}
#     for i in ${!seeds[@]}
#     do
#         seed=${seeds[$i]}
#         gpu_id=${gpus[$i]}
#         echo "Running on GPU $gpu_id with seed $seed"
#         python ../train_baseline_multi_modality.py \
#         --device_system=$device_system \
#         --seed=$seed \
#         --subject_id=$subject_id \
#         --backbone='DBConformer' \
#         --gpu_id=$gpu_id \
#         --channel_path='read_full' \
#         --out_dim 256 \
#         --out_time 32 \
#         --temperature 2 \
#         &
#         sleep 3
#     done

#     wait
#     echo "subject tasks completed"
# done

# wait
# echo "All tasks completed"


# #-------------------------------------------------------------------------------------read_full 1#

# for s in ${!subjects[@]}
# do
#     subject_id=${subjects[$s]}
#     device_system=${systems[$s]}
#     for i in ${!seeds[@]}
#     do
#         seed=${seeds[$i]}
#         gpu_id=${gpus[$i]}
#         echo "Running on GPU $gpu_id with seed $seed"
#         python ../train_baseline_multi_modality.py \
#         --device_system=$device_system \
#         --seed=$seed \
#         --subject_id=$subject_id \
#         --backbone='DBConformer' \
#         --gpu_id=$gpu_id \
#         --channel_path='read_full' \
#         --out_dim 256 \
#         --out_time 32 \
#         --temperature 1 \
#         &
#         sleep 3
#     done

#     wait
#     echo "subject tasks completed"
# done

# wait
# echo "All tasks completed"

# #-------------------------------------------------------------------------------------read_full hubertraw_0 #

# for s in ${!subjects[@]}
# do
#     subject_id=${subjects[$s]}
#     device_system=${systems[$s]}
#     for i in ${!seeds[@]}
#     do
#         seed=${seeds[$i]}
#         gpu_id=${gpus[$i]}
#         echo "Running on GPU $gpu_id with seed $seed"
#         python ../train_baseline_multi_modality.py \
#         --device_system=$device_system \
#         --seed=$seed \
#         --subject_id=$subject_id \
#         --backbone='DBConformer' \
#         --gpu_id=$gpu_id \
#         --channel_path='read_full' \
#         --feature_type='hubertraw_0' \
#         --out_dim 256 \
#         --out_time 32 \
#         &
#         sleep 3
#     done

#     wait
#     echo "subject tasks completed"
# done

# wait
# echo "All tasks completed"

# #-------------------------------------------------------------------------------------read_full hubertraw_2 #

# for s in ${!subjects[@]}
# do
#     subject_id=${subjects[$s]}
#     device_system=${systems[$s]}
#     for i in ${!seeds[@]}
#     do
#         seed=${seeds[$i]}
#         gpu_id=${gpus[$i]}
#         echo "Running on GPU $gpu_id with seed $seed"
#         python ../train_baseline_multi_modality.py \
#         --device_system=$device_system \
#         --seed=$seed \
#         --subject_id=$subject_id \
#         --backbone='DBConformer' \
#         --gpu_id=$gpu_id \
#         --channel_path='read_full' \
#         --feature_type='hubertraw_2' \
#         --out_dim 256 \
#         --out_time 32 \
#         &
#         sleep 3
#     done

#     wait
#     echo "subject tasks completed"
# done

# wait
# echo "All tasks completed"


# #-------------------------------------------------------------------------------------read_full hubertraw_4 #

# for s in ${!subjects[@]}
# do
#     subject_id=${subjects[$s]}
#     device_system=${systems[$s]}
#     for i in ${!seeds[@]}
#     do
#         seed=${seeds[$i]}
#         gpu_id=${gpus[$i]}
#         echo "Running on GPU $gpu_id with seed $seed"
#         python ../train_baseline_multi_modality.py \
#         --device_system=$device_system \
#         --seed=$seed \
#         --subject_id=$subject_id \
#         --backbone='DBConformer' \
#         --gpu_id=$gpu_id \
#         --channel_path='read_full' \
#         --feature_type='hubertraw_4' \
#         --out_dim 256 \
#         --out_time 32 \
#         &
#         sleep 3
#     done

#     wait
#     echo "subject tasks completed"
# done

# wait
# echo "All tasks completed"


# #-------------------------------------------------------------------------------------read_full hubertraw_6 #

# for s in ${!subjects[@]}
# do
#     subject_id=${subjects[$s]}
#     device_system=${systems[$s]}
#     for i in ${!seeds[@]}
#     do
#         seed=${seeds[$i]}
#         gpu_id=${gpus[$i]}
#         echo "Running on GPU $gpu_id with seed $seed"
#         python ../train_baseline_multi_modality.py \
#         --device_system=$device_system \
#         --seed=$seed \
#         --subject_id=$subject_id \
#         --backbone='DBConformer' \
#         --gpu_id=$gpu_id \
#         --channel_path='read_full' \
#         --feature_type='hubertraw_6' \
#         --out_dim 256 \
#         --out_time 32 \
#         &
#         sleep 3
#     done

#     wait
#     echo "subject tasks completed"
# done

# wait
# echo "All tasks completed"


# #-------------------------------------------------------------------------------------read_full hubertraw_8 #

# for s in ${!subjects[@]}
# do
#     subject_id=${subjects[$s]}
#     device_system=${systems[$s]}
#     for i in ${!seeds[@]}
#     do
#         seed=${seeds[$i]}
#         gpu_id=${gpus[$i]}
#         echo "Running on GPU $gpu_id with seed $seed"
#         python ../train_baseline_multi_modality.py \
#         --device_system=$device_system \
#         --seed=$seed \
#         --subject_id=$subject_id \
#         --backbone='DBConformer' \
#         --gpu_id=$gpu_id \
#         --channel_path='read_full' \
#         --feature_type='hubertraw_8' \
#         --out_dim 256 \
#         --out_time 32 \
#         &
#         sleep 3
#     done

#     wait
#     echo "subject tasks completed"
# done

# wait
# echo "All tasks completed"


# #-------------------------------------------------------------------------------------read_full hubertraw_10 #

# for s in ${!subjects[@]}
# do
#     subject_id=${subjects[$s]}
#     device_system=${systems[$s]}
#     for i in ${!seeds[@]}
#     do
#         seed=${seeds[$i]}
#         gpu_id=${gpus[$i]}
#         echo "Running on GPU $gpu_id with seed $seed"
#         python ../train_baseline_multi_modality.py \
#         --device_system=$device_system \
#         --seed=$seed \
#         --subject_id=$subject_id \
#         --backbone='DBConformer' \
#         --gpu_id=$gpu_id \
#         --channel_path='read_full' \
#         --feature_type='hubertraw_10' \
#         --out_dim 256 \
#         --out_time 32 \
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
# log->csv
#######################################################################################################
#######################################################################################################
#######################################################################################################

current_date=$(date +%Y-%m-%d)
python log2csv.py --today "$current_date" --seeds ${seeds[@]} --subjects ${subjects[@]} &

wait