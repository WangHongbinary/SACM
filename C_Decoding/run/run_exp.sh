#!/bin/bash

##########
# hubert_8
##########

seeds=(2024 2025 2026 2027 2028 2029)
gpus=(0 1 2 3 4 5)

subjects=('S01' 'S02' 'S03' 'S04' 'S05' 'S06' 'S07' 'S08')
systems=('Natus/V_48' 'Natus/V_48' 'Natus/V_48' 'Natus/V_48' 'Natus/V_48' 'Natus/V_48' 'Neuracle/V_48' 'Neuracle/V_48')

#------------------------------------------------------------------------------------read_full random #

for s in ${!subjects[@]}
do
    subject_id=${subjects[$s]}
    device_system=${systems[$s]}
    for i in ${!seeds[@]}
    do
        seed=${seeds[$i]}
        gpu_id=${gpus[$i]}
        echo "Running on GPU $gpu with seed $seed"
        python ../train.py --device_system=$device_system --seed=$seed --subject_id=$subject_id --gpu_id=$gpu_id --channel_path='read_full' --random=True &
        sleep 3
    done

    wait
    echo "subject tasks completed"
done 

wait
echo "All tasks completed"

# #------------------------------------------------------------------------------------read_single_A #

# for s in ${!subjects[@]}
# do
#     subject_id=${subjects[$s]}
#     device_system=${systems[$s]}
#     for i in ${!seeds[@]}
#     do
#         seed=${seeds[$i]}
#         gpu_id=${gpus[$i]}
#         echo "Running on GPU $gpu with seed $seed"
#         python ../train.py --device_system=$device_system --seed=$seed --subject_id=$subject_id --gpu_id=$gpu_id --channel_path='read_single_A' &
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
#         echo "Running on GPU $gpu with seed $seed"
#         python ../train.py --device_system=$device_system --seed=$seed --subject_id=$subject_id --gpu_id=$gpu_id --channel_path='read_single_B' &
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
#         echo "Running on GPU $gpu with seed $seed"
#         python ../train.py --device_system=$device_system --seed=$seed --subject_id=$subject_id --gpu_id=$gpu_id --channel_path='read_single_C' &
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
#         echo "Running on GPU $gpu with seed $seed"
#         python ../train.py --device_system=$device_system --seed=$seed --subject_id=$subject_id --gpu_id=$gpu_id --channel_path='read_single_D' &
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
#         echo "Running on GPU $gpu with seed $seed"
#         python ../train.py --device_system=$device_system --seed=$seed --subject_id=$subject_id --gpu_id=$gpu_id --channel_path='read_single_E' &
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
#         echo "Running on GPU $gpu with seed $seed"
#         python ../train.py --device_system=$device_system --seed=$seed --subject_id=$subject_id --gpu_id=$gpu_id --channel_path='read_single_F' &
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
#         echo "Running on GPU $gpu with seed $seed"
#         python ../train.py --device_system=$device_system --seed=$seed --subject_id=$subject_id --gpu_id=$gpu_id --channel_path='read_single_G' &
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
        echo "Running on GPU $gpu with seed $seed"
        python ../train.py --device_system=$device_system --seed=$seed --subject_id=$subject_id --gpu_id=$gpu_id --channel_path='read_full' &
        sleep 3
    done

    wait
    echo "subject tasks completed"
done 

wait
echo "All tasks completed"


# #------------------------------------------------------------------------------------read_full_bipolar random#

# for s in ${!subjects[@]}
# do
#     subject_id=${subjects[$s]}
#     device_system=${systems[$s]}
#     for i in ${!seeds[@]}
#     do
#         seed=${seeds[$i]}
#         gpu_id=${gpus[$i]}
#         echo "Running on GPU $gpu with seed $seed"
#         python ../train.py --device_system=$device_system --seed=$seed --subject_id=$subject_id --gpu_id=$gpu_id --channel_path='read_full_bipolar' --random=True &
#         sleep 3
#     done

#     wait
#     echo "subject tasks completed"
# done 

# wait
# echo "All tasks completed"

# #------------------------------------------------------------------------------------read_single_A_bipolar #

# for s in ${!subjects[@]}
# do
#     subject_id=${subjects[$s]}
#     device_system=${systems[$s]}
#     for i in ${!seeds[@]}
#     do
#         seed=${seeds[$i]}
#         gpu_id=${gpus[$i]}
#         echo "Running on GPU $gpu with seed $seed"
#         python ../train.py --device_system=$device_system --seed=$seed --subject_id=$subject_id --gpu_id=$gpu_id --channel_path='read_single_A_bipolar' &
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
#         echo "Running on GPU $gpu with seed $seed"
#         python ../train.py --device_system=$device_system --seed=$seed --subject_id=$subject_id --gpu_id=$gpu_id --channel_path='read_single_B_bipolar' &
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
#         echo "Running on GPU $gpu with seed $seed"
#         python ../train.py --device_system=$device_system --seed=$seed --subject_id=$subject_id --gpu_id=$gpu_id --channel_path='read_single_C_bipolar' &
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
#         echo "Running on GPU $gpu with seed $seed"
#         python ../train.py --device_system=$device_system --seed=$seed --subject_id=$subject_id --gpu_id=$gpu_id --channel_path='read_single_D_bipolar' &
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
#         echo "Running on GPU $gpu with seed $seed"
#         python ../train.py --device_system=$device_system --seed=$seed --subject_id=$subject_id --gpu_id=$gpu_id --channel_path='read_single_E_bipolar' &
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
#         echo "Running on GPU $gpu with seed $seed"
#         python ../train.py --device_system=$device_system --seed=$seed --subject_id=$subject_id --gpu_id=$gpu_id --channel_path='read_single_F_bipolar' &
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
#         echo "Running on GPU $gpu with seed $seed"
#         python ../train.py --device_system=$device_system --seed=$seed --subject_id=$subject_id --gpu_id=$gpu_id --channel_path='read_single_G_bipolar' &
#         sleep 3
#     done

#     wait
#     echo "subject tasks completed"
# done 

# wait
# echo "All tasks completed"

#-------------------------------------------------------------------------------------read_full_bipolar #

for s in ${!subjects[@]}
do
    subject_id=${subjects[$s]}
    device_system=${systems[$s]}
    for i in ${!seeds[@]}
    do
        seed=${seeds[$i]}
        gpu_id=${gpus[$i]}
        echo "Running on GPU $gpu with seed $seed"
        python ../train.py --device_system=$device_system --seed=$seed --subject_id=$subject_id --gpu_id=$gpu_id --channel_path='read_full_bipolar' &
        sleep 3
    done

    wait
    echo "subject tasks completed"
done 

wait
echo "All tasks completed"





################################
# Layer Depth Analysis of HuBERT
################################

# seeds=(2024 2025 2026 2027 2028 2029)
# gpus=(0 1 2 3 4 5)
# features=('hubert_0' 'hubert_1' 'hubert_2' 'hubert_3' 'hubert_4' 'hubert_5' 'hubert_6' 'hubert_7' 'hubert_8' 'hubert_9' 'hubert_10' 'hubert_11')
# subjects=('S01' 'S02' 'S03' 'S04' 'S05' 'S06' 'S07' 'S08')
# systems=('Natus/V_48' 'Natus/V_48' 'Natus/V_48' 'Natus/V_48' 'Natus/V_48' 'Natus/V_48' 'Neuracle/V_48' 'Neuracle/V_48')

# # read_full
# for f in ${!features[@]}
# do
#     feature_type=${features[$f]}
#     for s in ${!subjects[@]}
#     do
#         subject_id=${subjects[$s]}
#         device_system=${systems[$s]}
#         for i in ${!seeds[@]}
#         do
#             seed=${seeds[$i]}
#             gpu_id=${gpus[$i]}
#             echo "Running on GPU $gpu_id with seed $seed and feature $feature_type"
#             python ../train.py --device_system=$device_system --seed=$seed --subject_id=$subject_id --gpu_id=$gpu_id --feature_type=$feature_type --channel_path='read_full' &
#             sleep 3
#         done

#         wait
#         echo "subject tasks completed"
#     done 

#     wait
#     echo "feature_type tasks completed"
# done

# wait
# echo "All tasks completed"


# # read_full_bipolar
# for f in ${!features[@]}
# do
#     feature_type=${features[$f]}
#     for s in ${!subjects[@]}
#     do
#         subject_id=${subjects[$s]}
#         device_system=${systems[$s]}
#         for i in ${!seeds[@]}
#         do
#             seed=${seeds[$i]}
#             gpu_id=${gpus[$i]}
#             echo "Running on GPU $gpu_id with seed $seed and feature $feature_type"
#             python ../train.py --device_system=$device_system --seed=$seed --subject_id=$subject_id --gpu_id=$gpu_id --feature_type=$feature_type --channel_path='read_full_bipolar' &
#             sleep 3
#         done

#         wait
#         echo "subject tasks completed"
#     done 

#     wait
#     echo "feature_type tasks completed"
# done

# wait
# echo "All tasks completed"





##########
# log->csv
##########

current_date=$(date +%Y-%m-%d)
python log2csv.py --today "$current_date" --seeds ${seeds[@]} --subjects ${subjects[@]} &

wait